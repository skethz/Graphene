"""Model loaders for the super-resolution cost benchmark (Params, FLOPs, latency, memory).

One loader per method builds the network from the authors' official repository,
cloned at the commit pinned in ``third_party.json``. Nothing from those
repositories is copied into this package. Each loader returns a ``Loaded``
record with the model on the CPU, the padding multiple used by the protocol,
and a description of the weights.

Only the import plumbing below is ours:

* ``_repo_basicsr`` exposes one repository's own ``basicsr/`` directory as the
  ``basicsr`` package without running ``basicsr/__init__.py``. That file
  imports the training stack (data, losses, models), which inference does not
  need. Each repository therefore uses its own BasicSR copy.
* ``_basicsr_shim`` is used only for HAT, whose architecture file imports the
  pip ``basicsr`` package, and for UCAN (registry only). It provides
  ``ARCH_REGISTRY`` (a no-op decorator) and ``to_2tuple``/``trunc_normal_``
  (from timm). These functions only run at construction and initialisation,
  and the official weights then overwrite every parameter.
* ``_fairscale_shim`` provides ``fairscale.nn.checkpoint_wrapper`` for PFT and
  IET. Both repositories apply it only to layers with
  ``idx < idx_checkpoint == 0``, i.e. never. Under no-grad inference the
  original wrapper would be a pass-through anyway.

Five lightweight baselines have no public checkpoint at the pinned commits
(SwinIR-light, SRFormer-light, ATD-light, MambaIR-light, MambaIRv2-light).
They are built with random initialisation after ``torch.manual_seed(2447)``,
and their exact parameter counts are asserted. Cost does not depend on the
weight values for these architectures.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent

CLASSICAL = [
    "SwinIR", "CAT-A", "ART", "HAT", "SRFormer", "IPG", "ATD",
    "MambaIR", "MambaIRv2", "PFT", "IET", "Graphene",
]
LIGHTWEIGHT = [
    "SwinIR-light", "SRFormer-light", "ATD-light", "MambaIR-light", "MambaIRv2-light",
    "PFT-light", "UCAN", "IET-light", "Graphene-light",
]
METHODS = CLASSICAL + LIGHTWEIGHT
SCALES = (2, 3, 4)

# Exact parameter counts of the random-initialised baselines (Tables 13/14).
RANDOM_INIT_PARAMS = {
    "SwinIR-light": {2: 910_152, 3: 918_267, 4: 929_628},
    "SRFormer-light": {2: 853_272, 3: 861_387, 4: 872_748},
    "ATD-light": {2: 760_956, 3: 767_451, 4: 776_544},
    "MambaIR-light": {2: 905_016, 3: 913_131, 4: 924_492},
    "MambaIRv2-light": {2: 774_092, 3: 780_587, 4: 789_680},
}

# NATTEN kernel selection for IET (checked against a float64 reference in verify/natten_check.py):
#   fp32           NATTEN's non-GEMM CUDA kernels (natten.use_gemm_na(False)); true FP32, max error ~1e-6.
#                  Default, because the protocol turns TF32 off.
#   natten_default NATTEN's defaults: GEMM-based kernels with TF32 (max error ~5e-3 on SM90).
#   gemm_tf32_off  GEMM-based kernels after natten.use_tf32_in_gemm_na(False); still ~2e-3 on SM90.
NATTEN_MODES = ("fp32", "natten_default", "gemm_tf32_off")

# Padding multiple of the protocol (right/bottom flip padding, output cropped back).
# Window size of each architecture; IPG uses lcm(window 16, sample_size 32).
PROTOCOL_WINDOW = {
    "SwinIR": 8, "CAT-A": 16, "ART": 8, "HAT": 16, "SRFormer": 22, "IPG": 32, "ATD": 16,
    "MambaIR": 16, "MambaIRv2": 16, "PFT": 32, "IET": 32, "Graphene": 32,
    "SwinIR-light": 8, "SRFormer-light": 16, "ATD-light": 16, "MambaIR-light": 16, "MambaIRv2-light": 16,
    "PFT-light": 32, "UCAN": 16, "IET-light": 32, "Graphene-light": 16,
}


def _variant(scale: int) -> str:
    """Official config/checkpoint suffix: x2 is trained from scratch, x3/x4 are fine-tuned from x2."""
    return "scratch" if scale == 2 else "finetune"


def default_graphene_root(third_party: Path) -> Path:
    """third_party/Graphene if present, else the repository that contains this package (the Graphene release)."""
    if (third_party / "Graphene" / "models" / "graphene.py").is_file():
        return third_party / "Graphene"
    return PACKAGE_ROOT


@dataclass
class Roots:
    third_party: Path
    checkpoints: Path
    build: Path
    graphene: Path

    @classmethod
    def from_args(cls, third_party=None, checkpoints=None, build=None, graphene=None) -> "Roots":
        tp = Path(third_party or os.environ.get("SRBENCH_THIRD_PARTY", PACKAGE_ROOT / "third_party")).resolve()
        return cls(
            third_party=tp,
            checkpoints=Path(checkpoints or os.environ.get("SRBENCH_CHECKPOINTS", PACKAGE_ROOT / "checkpoints")).resolve(),
            build=Path(build or os.environ.get("SRBENCH_BUILD", PACKAGE_ROOT / "build")).resolve(),
            graphene=Path(graphene or os.environ.get("SRBENCH_GRAPHENE") or default_graphene_root(tp)).resolve(),
        )


@dataclass
class Loaded:
    method: str
    scale: int
    model: torch.nn.Module
    window: int
    weights: str  # "official checkpoint" or "random init (seed 2447)"
    repo: dict[str, Any] | None
    config: str | None
    checkpoint: dict[str, Any] | None = None
    prepare: Callable[[torch.nn.Module, int, int], None] | None = None
    extensions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def third_party_spec() -> dict[str, Any]:
    return json.loads((BENCH_DIR / "third_party.json").read_text())["repos"]


def checkpoint_spec() -> dict[str, Any]:
    return json.loads((BENCH_DIR / "checkpoints.json").read_text())["checkpoints"]


def repo_state(roots: Roots, name: str) -> dict[str, Any]:
    """Commit of a third-party clone, compared with the pinned commit."""
    spec = third_party_spec()[name]
    path = roots.third_party / name
    state: dict[str, Any] = {"name": name, "url": spec["url"], "pinned": spec["commit"], "path": str(path)}
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.split("\n")
        state.update(commit=head, matches_pin=head == spec["commit"],
                     modified_tracked_files=[line[3:] for line in dirty if line.strip()])
    except Exception as exc:  # not a git checkout
        state.update(commit=None, matches_pin=False, error=str(exc))
    return state


def _purge(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _import_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torchvision_compat() -> None:
    """torchvision>=0.17 removed transforms.functional_tensor, which old BasicSR copies import."""
    try:
        import torchvision.transforms.functional as functional
    except Exception:
        return
    compat = types.ModuleType("torchvision.transforms.functional_tensor")
    compat.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules.setdefault(compat.__name__, compat)


def _repo_basicsr(repo: Path) -> None:
    """Expose <repo>/basicsr as package `basicsr` without executing its __init__.py."""
    _purge("basicsr")
    namespace = types.ModuleType("basicsr")
    namespace.__path__ = [str(repo / "basicsr")]
    namespace.__package__ = "basicsr"
    sys.modules["basicsr"] = namespace
    version = types.ModuleType("basicsr.version")  # generated by BasicSR's setup.py, absent in clones
    version.__version__, version.__gitsha__ = "0.0.0", "unknown"
    sys.modules["basicsr.version"] = version
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _torchvision_compat()


def _basicsr_shim() -> None:
    """Minimal stand-in for the pip `basicsr` names imported by HAT and UCAN."""
    from timm.layers import to_2tuple, trunc_normal_

    class _Registry:
        def register(self, *args, **kwargs):
            def decorator(obj):
                return obj
            if args and callable(args[0]):
                return decorator(args[0])
            return decorator

    _purge("basicsr")
    modules = {name: types.ModuleType(name) for name in (
        "basicsr", "basicsr.utils", "basicsr.utils.registry", "basicsr.archs", "basicsr.archs.arch_util")}
    for name in ("basicsr", "basicsr.utils", "basicsr.archs"):
        modules[name].__path__ = []
    modules["basicsr.utils.registry"].ARCH_REGISTRY = _Registry()
    modules["basicsr.archs.arch_util"].to_2tuple = to_2tuple
    modules["basicsr.archs.arch_util"].trunc_normal_ = trunc_normal_
    modules["basicsr"].utils = modules["basicsr.utils"]
    modules["basicsr"].archs = modules["basicsr.archs"]
    modules["basicsr.utils"].registry = modules["basicsr.utils.registry"]
    modules["basicsr.archs"].arch_util = modules["basicsr.archs.arch_util"]
    sys.modules.update(modules)
    _torchvision_compat()


def _fairscale_shim() -> None:
    fairscale = types.ModuleType("fairscale")
    fairscale_nn = types.ModuleType("fairscale.nn")
    fairscale_nn.checkpoint_wrapper = lambda module, *args, **kwargs: module
    fairscale.nn = fairscale_nn
    sys.modules["fairscale"] = fairscale
    sys.modules["fairscale.nn"] = fairscale_nn


def _smm_cuda(roots: Roots, repo_name: str) -> dict[str, Any]:
    """Import the smm_cuda extension built from <repo>/ops_smm (PFT-SR and IET ship different sources)."""
    _purge("smm_cuda")
    directory = roots.build / "smm" / repo_name
    libraries = sorted(directory.glob("smm_cuda*.so"))
    if not libraries:
        raise FileNotFoundError(f"{directory}: smm_cuda not built; run benchmarks/build_smm.py --repo {repo_name}")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module("smm_cuda")
    if Path(module.__file__).resolve() != libraries[0].resolve():
        raise ImportError(f"smm_cuda resolved to {module.__file__}, expected {libraries[0]}")
    build_info = directory / "build_info.json"
    return {"smm_cuda": {"path": str(libraries[0]), "sha256": sha256_file(libraries[0]),
                         "build": json.loads(build_info.read_text()) if build_info.exists() else None}}


def _natten(roots: Roots, mode: str) -> dict[str, Any]:
    """Import NATTEN built under <build>/natten and select its kernel family."""
    if mode not in NATTEN_MODES:
        raise ValueError(f"natten mode {mode!r} not in {NATTEN_MODES}")
    directory = roots.build / "natten"
    if not (directory / "natten").is_dir():
        raise FileNotFoundError(f"{directory}: NATTEN not built; see setup_third_party.sh")
    _purge("natten")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import natten
    from natten import libnatten

    if not str(Path(natten.__file__).resolve()).startswith(str(directory.resolve())):
        raise ImportError(f"natten resolved to {natten.__file__}, expected under {directory}")
    if mode == "fp32":
        natten.use_gemm_na(False)
    elif mode == "gemm_tf32_off":
        natten.use_tf32_in_gemm_na(False)
    library = Path(libnatten.__file__)
    return {"natten": {
        "version": natten.__version__, "path": str(directory), "libnatten": str(library),
        "libnatten_sha256": sha256_file(library), "mode": mode,
        "has_cuda": natten.has_cuda(), "has_gemm": natten.has_gemm(),
        "gemm_na": natten.is_gemm_na_enabled(), "tiled_na": natten.is_tiled_na_enabled(),
        "tf32_in_gemm_na": natten.is_tf32_in_gemm_na_enabled(), "fused_na": natten.is_fused_na_enabled(),
    }}


def _select_state(checkpoint: Any, preferred: str | None = None) -> tuple[dict, str]:
    if not isinstance(checkpoint, dict):
        return checkpoint, "<raw>"
    for key in (preferred, "params_ema", "params", "state_dict", "net_g"):
        if key and key in checkpoint:
            return checkpoint[key], key
    return checkpoint, "<dict>"


def _checkpoint(roots: Roots, method: str, scale: int, verify: bool = True) -> tuple[Path, dict[str, Any]]:
    entry = checkpoint_spec()[method][str(scale)]
    path = roots.checkpoints / entry["file"]
    if not path.is_file() and (roots.checkpoints / Path(entry["file"]).name).is_file():
        path = roots.checkpoints / Path(entry["file"]).name  # flat layout, e.g. the Graphene repository's checkpoints/
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing official checkpoint for {method} x{scale}; source: {entry['source']}")
    digest = sha256_file(path)
    record = {"file": entry["file"], "path": str(path), "resolved": str(path.resolve()), "bytes": path.stat().st_size,
              "sha256": digest, "expected_sha256": entry["sha256"], "sha256_ok": digest == entry["sha256"]}
    if verify and digest != entry["sha256"]:
        raise RuntimeError(f"{path}: sha256 {digest} != expected {entry['sha256']}")
    return path, record


def _load_state(path: Path, preferred: str | None) -> tuple[dict, str]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Some official BasicSR checkpoints pickle non-tensor metadata; the file's sha256 was checked first.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return _select_state(checkpoint, preferred)


def _strict(model: torch.nn.Module, state: dict, label: str, allow_missing_suffix: str | None = None) -> None:
    result = model.load_state_dict(state, strict=allow_missing_suffix is None)
    if allow_missing_suffix is not None:
        missing = [k for k in result.missing_keys if not k.endswith(allow_missing_suffix)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"{label}: checkpoint mismatch missing={missing[:8]} unexpected={result.unexpected_keys[:8]}")


def _network(config: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(config.read_text())["network_g"])


def _official(roots: Roots, method: str, scale: int, model: torch.nn.Module, config: Path | None,
              preferred_key: str | None, verify: bool, allow_missing_suffix: str | None = None):
    path, record = _checkpoint(roots, method, scale, verify)
    state, key = _load_state(path, preferred_key)
    _strict(model, state, f"{method} x{scale}", allow_missing_suffix)
    record["state_key"] = key
    if allow_missing_suffix:
        record["missing_allowed"] = f"buffers ending in {allow_missing_suffix} (rebuilt by the constructor)"
    return record


def _random_init(method: str, scale: int, model: torch.nn.Module) -> None:
    params = sum(p.numel() for p in model.parameters())
    expected = RANDOM_INIT_PARAMS[method][scale]
    if params != expected:
        raise RuntimeError(f"{method} x{scale}: {params:,} parameters, expected {expected:,}")


# --------------------------------------------------------------------------- loaders

def _swinir(roots: Roots, method: str, scale: int, verify: bool) -> Loaded:
    repo = roots.third_party / "SwinIR"
    module = _import_file("srbench_swinir", repo / "models/network_swinir.py")
    light = method == "SwinIR-light"
    # Arguments of main_test_swinir.py define_model(): task classical_sr (training_patch_size 64) / lightweight_sr.
    kwargs = dict(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                  depths=[6] * (4 if light else 6), embed_dim=60 if light else 180,
                  num_heads=[6] * (4 if light else 6), mlp_ratio=2,
                  upsampler="pixelshuffledirect" if light else "pixelshuffle", resi_connection="1conv")
    model = module.SwinIR(**kwargs)
    loaded = Loaded(method, scale, model, 8, "", repo_state(roots, "SwinIR"),
                    "main_test_swinir.py define_model(" + ("lightweight_sr" if light else "classical_sr") + ")")
    if light:
        _random_init(method, scale, model)
        loaded.weights = "random init (seed 2447)"
    else:
        loaded.checkpoint = _official(roots, method, scale, model, None, "params", verify)
        loaded.weights = "official checkpoint"
    return loaded


def _basicsr_build(roots: Roots, method: str, scale: int, repo_name: str, subdir: str, config_rel: str,
                   verify: bool, random_init: bool = False) -> Loaded:
    """Repositories whose test path is `basicsr.archs.build_network(opt['network_g'])`."""
    repo = roots.third_party / repo_name / subdir if subdir else roots.third_party / repo_name
    _repo_basicsr(repo)
    from basicsr.archs import build_network

    config = repo / config_rel
    opt = yaml.safe_load(config.read_text())
    network = opt["network_g"]
    model = build_network(dict(network))
    window = int(network.get("window_size", 16))
    window = math.lcm(window, int(network.get("sample_size", window)))
    loaded = Loaded(method, scale, model, window, "", repo_state(roots, repo_name), str(config.relative_to(roots.third_party)))
    if random_init:
        _random_init(method, scale, model)
        loaded.weights = "random init (seed 2447)"
    else:
        loaded.checkpoint = _official(roots, method, scale, model, config,
                                      opt.get("path", {}).get("param_key_g"), verify)
        loaded.weights = "official checkpoint"
    if network.get("use_checkpoint"):
        loaded.notes.append("official config sets use_checkpoint: true (kept); it only matters with autograd enabled")
    return loaded


def _class_from_config(roots: Roots, method: str, scale: int, repo_name: str, module_name: str,
                       class_name: str, config_rel: str, verify: bool, random_init: bool,
                       window: int | None = None) -> Loaded:
    """Import one architecture module from the repository's basicsr copy and build it from the test yml."""
    repo = roots.third_party / repo_name
    _repo_basicsr(repo)
    module = importlib.import_module(f"basicsr.archs.{module_name}")
    config = repo / config_rel
    opt = yaml.safe_load(config.read_text())
    network = dict(opt["network_g"])
    network.pop("type", None)
    model = getattr(module, class_name)(**network)
    loaded = Loaded(method, scale, model, window or int(network.get("window_size", 16)), "",
                    repo_state(roots, repo_name), str(config.relative_to(roots.third_party)))
    if random_init:
        _random_init(method, scale, model)
        loaded.weights = "random init (seed 2447)"
    else:
        loaded.checkpoint = _official(roots, method, scale, model, config,
                                      opt.get("path", {}).get("param_key_g"), verify)
        loaded.weights = "official checkpoint"
    return loaded


def _hat(roots: Roots, method: str, scale: int, verify: bool) -> Loaded:
    repo = roots.third_party / "HAT"
    _basicsr_shim()
    module = _import_file("srbench_hat_arch", repo / "hat/archs/hat_arch.py")
    config = repo / f"options/test/HAT_SRx{scale}.yml"
    opt = yaml.safe_load(config.read_text())
    network = dict(opt["network_g"])
    network.pop("type", None)
    model = module.HAT(**network)
    loaded = Loaded(method, scale, model, int(network.get("window_size", 16)), "official checkpoint",
                    repo_state(roots, "HAT"), str(config.relative_to(roots.third_party)))
    loaded.checkpoint = _official(roots, method, scale, model, config, opt["path"].get("param_key_g"), verify)
    loaded.notes.append("basicsr registry/arch_util names provided by timm (HAT imports the pip basicsr package)")
    return loaded


def _pft(roots: Roots, method: str, scale: int, verify: bool) -> Loaded:
    repo = roots.third_party / "PFT-SR"
    _repo_basicsr(repo)
    _fairscale_shim()
    extensions = _smm_cuda(roots, "PFT-SR")
    prefix = "10" if method == "PFT-light" else "00"
    kind = "PFT_light" if method == "PFT-light" else "PFT"
    config = repo / f"options/test/{prefix}{scale - 1}_{kind}_SRx{scale}_{_variant(scale)}.yml"
    opt = yaml.safe_load(config.read_text())
    network = dict(opt["network_g"])
    network.pop("type", None)
    module = importlib.import_module("basicsr.archs.pft_arch")
    model = module.PFT(**network)
    loaded = Loaded(method, scale, model, 32, "official checkpoint", repo_state(roots, "PFT-SR"),
                    str(config.relative_to(roots.third_party)), extensions=extensions)
    loaded.checkpoint = _official(roots, method, scale, model, config, opt["path"].get("param_key_g"), verify)
    return loaded


def _iet(roots: Roots, method: str, scale: int, verify: bool, natten_mode: str) -> Loaded:
    repo = roots.third_party / "IET"
    _repo_basicsr(repo)
    _fairscale_shim()
    extensions = _smm_cuda(roots, "IET")
    extensions.update(_natten(roots, natten_mode))
    if method == "IET-light":
        config = repo / f"options/test/10{scale - 1}_IET_light_SRx{scale}_{_variant(scale)}.yml"
    else:
        config = repo / f"options/test/00{scale - 1}_IET_SRx{scale}_{_variant(scale)}.yml"
    opt = yaml.safe_load(config.read_text())
    network = dict(opt["network_g"])
    network.pop("type", None)
    module = importlib.import_module("basicsr.archs.iet_arch")
    model = module.IET(**network)
    loaded = Loaded(method, scale, model, 32, "official checkpoint", repo_state(roots, "IET"),
                    str(config.relative_to(roots.third_party)), extensions=extensions)
    # The released IET checkpoints do not contain the relative_position_index buffers; the constructor builds them.
    loaded.checkpoint = _official(roots, method, scale, model, config, opt["path"].get("param_key_g"), verify,
                                  allow_missing_suffix="relative_position_index")
    loaded.notes.append("IET.forward flip-pads internally to a multiple of sparse_range*dilation; "
                        "the protocol's external padding to 32 is a no-op at 64x64")
    return loaded


def _ucan(roots: Roots, method: str, scale: int, verify: bool) -> Loaded:
    repo = roots.third_party / "UCAN"
    _basicsr_shim()
    module = _import_file("srbench_ucan_arch", repo / "basicsr/archs/ucan_arch.py")
    config = repo / f"options/Test/test_UCAN_x{scale}.yml"
    opt = yaml.safe_load(config.read_text())
    network = dict(opt["network_g"])
    network.pop("type", None)
    model = module.UCAN(**network)
    loaded = Loaded(method, scale, model, int(network.get("window_size", 16)), "official checkpoint",
                    repo_state(roots, "UCAN"), str(config.relative_to(roots.third_party)))
    loaded.checkpoint = _official(roots, method, scale, model, config, opt["path"].get("param_key_g", "params_ema"), verify)
    return loaded


def graphene_code_state(root: Path, config: Path) -> dict[str, Any]:
    files = sorted(p for p in (root / "models").rglob("*.py"))
    entries = {str(p.relative_to(root)): sha256_file(p) for p in files}
    entries[str(config.relative_to(root))] = sha256_file(config)
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    state: dict[str, Any] = {"name": "Graphene", "path": str(root), "files_sha256": entries, "tree_sha256": digest}
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        state["commit"] = head.stdout.strip()
    except Exception:
        state["commit"] = None
    return state


def _graphene(roots: Roots, method: str, scale: int, verify: bool) -> Loaded:
    root = roots.graphene
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _purge("models")
    from models.build import build_model, prepare_masks, read_config

    families = ("graphene-classical", "graphene") if method == "Graphene" else ("graphene-lightweight", "graphene-light")
    config_path = next((root / "configs" / f"{fam}_x{scale}.json" for fam in families
                        if (root / "configs" / f"{fam}_x{scale}.json").is_file()), root / "configs" / f"{families[0]}_x{scale}.json")
    config = read_config(config_path)
    path, record = _checkpoint(roots, method, scale, verify)
    model = build_model(config, path)  # strict load (attention masks rebuilt for the configured size)
    # FP32 input/output convolutions when autocast is on (BF16 mode); no effect in FP32 eager.
    model.fp32_io = True
    record["state_key"] = "params_ema/params (models.build.load_weights, strict)"
    window = int(config["model"]["window_size"])

    def prepare(m: torch.nn.Module, height: int, width: int) -> None:
        prepare_masks(m, height, width)

    return Loaded(method, scale, model, window, "official checkpoint", graphene_code_state(root, config_path),
                  str(config_path), record, prepare=prepare,
                  notes=["prepare_masks(model, H_pad, W_pad) caches shift masks for the measured size"])


def load(method: str, scale: int, roots: Roots, natten_mode: str = "fp32", verify_checkpoints: bool = True) -> Loaded:
    """Build `method` at `scale` on the CPU. Seed the RNG before calling for reproducible random init."""
    loaded = _load(method, scale, roots, natten_mode, verify_checkpoints)
    if loaded.window != PROTOCOL_WINDOW[method]:
        raise RuntimeError(f"{method}: loader window {loaded.window} != protocol window {PROTOCOL_WINDOW[method]}")
    loaded.model.eval()
    return loaded


def _load(method: str, scale: int, roots: Roots, natten_mode: str, verify_checkpoints: bool) -> Loaded:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    v, s = verify_checkpoints, scale
    if method in ("SwinIR", "SwinIR-light"):
        return _swinir(roots, method, s, v)
    if method == "CAT-A":
        return _basicsr_build(roots, method, s, "CAT", "", f"options/test/test_CAT_A_sr_x{s}.yml", v)
    if method == "ART":
        return _basicsr_build(roots, method, s, "ART", "", f"options/test/test_ART_SR_x{s}.yml", v)
    if method == "IPG":
        return _basicsr_build(roots, method, s, "Efficient-Computing", "LowLevel/IPG",
                              f"options/test_IPG_BasicSR_x{s}.yml", v)
    if method == "ATD":
        return _basicsr_build(roots, method, s, "ATD", "", f"options/test/sr/00{s - 1}_ATD_SRx{s}_finetune.yml", v)
    if method == "ATD-light":
        return _basicsr_build(roots, method, s, "ATD", "",
                              f"options/test/sr/10{s - 1}_ATD_light_SRx{s}_{_variant(s)}.yml", v, random_init=True)
    if method == "HAT":
        return _hat(roots, method, s, v)
    if method == "SRFormer":
        return _class_from_config(roots, method, s, "SRFormer", "srformer_arch", "SRFormer",
                                  f"options/test/SRFormer/test_SRFormer_DF2Ksrx{s}.yml", v, False)
    if method == "SRFormer-light":
        return _class_from_config(roots, method, s, "SRFormer", "srformer_arch", "SRFormer",
                                  f"options/test/SRFormer/test_SRFormer_light_DIV2Ksrx{s}.yml", v, True)
    if method == "MambaIR":
        return _class_from_config(roots, method, s, "MambaIR", "mambair_arch", "MambaIR",
                                  f"options/test/mambair/test_MambaIR_SR_x{s}.yml", v, False, window=16)
    if method == "MambaIR-light":
        return _class_from_config(roots, method, s, "MambaIR", "mambair_arch", "MambaIR",
                                  f"options/test/mambair/test_MambaIR_lightSR_x{s}.yml", v, True, window=16)
    if method == "MambaIRv2":
        return _class_from_config(roots, method, s, "MambaIR", "mambairv2_arch", "MambaIRv2",
                                  f"options/test/mambairv2/test_MambaIRv2_SR_x{s}.yml", v, False)
    if method == "MambaIRv2-light":
        return _class_from_config(roots, method, s, "MambaIR", "mambairv2light_arch", "MambaIRv2Light",
                                  f"options/test/mambairv2/test_MambaIRv2_lightSR_x{s}.yml", v, True)
    if method in ("PFT", "PFT-light"):
        return _pft(roots, method, s, v)
    if method in ("IET", "IET-light"):
        return _iet(roots, method, s, v, natten_mode)
    if method == "UCAN":
        return _ucan(roots, method, s, v)
    if method in ("Graphene", "Graphene-light"):
        return _graphene(roots, method, s, v)
    raise AssertionError(method)


# --------------------------------------------------------------------------- BF16 compiled fallbacks (11-15)

FALLBACKS = ("none", "smm_fp32", "smm_fp16", "cat_reshape", "cat_eager", "ipg_const", "ipg_eager", "ipg_cache")


def apply_fallback(loaded: "Loaded", name: str) -> dict[str, Any]:
    """Change only the part of a released model that cannot run BF16 autocast with torch.compile.

    smm_fp32 / smm_fp16 (IET, IET-light, PFT, PFT-light): the released sparse-matrix CUDA operators SMM_QmK and
        SMM_AmV accept FP32 (IET also FP16) but not BF16. For these two calls only, autocast is disabled, the inputs
        are cast to FP32 (or FP16), and the result is cast back to the dtype of the second input. All other
        operations keep BF16 autocast and compilation.
    cat_reshape (CAT-A): Inductor's lowering of torch.roll returns non-contiguous strides, so the released
        `.view(...)` calls right after torch.roll in the shifted blocks (CATB_axial and CATB_regular) fail to
        compile. These four calls per block become `.reshape(...)`, which returns the same values (a copy where a
        view is impossible).
    cat_eager (CAT-A): the released block forwards containing those calls run eagerly (torch.compiler.disable);
        the rest of the model is compiled.
    ipg_const (IPG): the released MGB.calc_graph creates two constant 3x3 tensors on the first forward and caches them
        on the module (self.tensors). With CUDA graphs (max-autotune) that cache is a CUDA-graph output, which the next
        replay overwrites. The same two constants are created on the GPU before compilation; nothing else changes.
    ipg_eager (IPG): the released graph construction MGB.calc_graph runs eagerly (torch.compiler.disable); the rest
        of the model is compiled.
    ipg_cache (IPG): in eval mode the released code fills two caches on its first forward (MGB.tensors, the constant
        graph-construction matrices, and IPG_Grapher.relative_position_bias_table, the output of its position-bias
        MLP). Under CUDA graphs both caches are CUDA-graph outputs that the next replay overwrites. One eager forward
        pass, with the same autocast setting, fills them before the first compiled call; the code is unchanged.
    """
    import inspect
    import textwrap
    if name == "none":
        return {"name": name}
    module = inspect.getmodule(type(loaded.model))
    if name in ("smm_fp32", "smm_fp16"):
        if not (hasattr(module, "SMM_QmK") and hasattr(module, "SMM_AmV")):
            raise ValueError(f"{name}: {loaded.method} has no SMM_QmK/SMM_AmV operators")
        dtype = torch.float32 if name == "smm_fp32" else torch.float16
        for op in ("SMM_QmK", "SMM_AmV"):
            original = getattr(module, op)
            if getattr(original, "_fallback", None):
                continue

            class _Fallback:
                _fallback = name
                _original = original

                @staticmethod
                def apply(a, b, index, _op=original, _dtype=dtype):
                    with torch.autocast("cuda", enabled=False):
                        y = _op.apply(a.to(_dtype), b.to(_dtype), index)
                    return y.to(b.dtype)

            setattr(module, op, _Fallback)
        return {"name": name, "operators": ["SMM_QmK", "SMM_AmV"], "dtype": str(dtype)}
    if name in ("cat_reshape", "cat_eager"):
        if loaded.method != "CAT-A":
            raise ValueError(f"{name} applies to CAT-A only")
        needle = "qkv_0 = qkv_0.view(3, B, L, C//2)"
        owners = sorted({type(m) for m in loaded.model.modules() if "forward" in type(m).__dict__
                         and needle in inspect.getsource(type(m).__dict__["forward"])}, key=lambda c: c.__name__)
        if not owners:
            raise ValueError(f"{name}: no block with {needle!r}")
        if name == "cat_eager":
            for cls in owners:
                cls.forward = torch.compiler.disable(cls.__dict__["forward"])
            return {"name": name, "classes": [c.__name__ for c in owners]}
        pairs = [("qkv_0 = qkv_0.view(3, B, L, C//2)", "qkv_0 = qkv_0.reshape(3, B, L, C//2)"),
                 ("qkv_1 = qkv_1.view(3, B, L, C//2)", "qkv_1 = qkv_1.reshape(3, B, L, C//2)"),
                 ("x1 = x1.view(B, L, C//2).contiguous()", "x1 = x1.reshape(B, L, C//2).contiguous()"),
                 ("x2 = x2.view(B, L, C//2).contiguous()", "x2 = x2.reshape(B, L, C//2).contiguous()")]
        for cls in owners:
            source = textwrap.dedent(inspect.getsource(cls.__dict__["forward"]))
            for old, new in pairs:
                if source.count(old) != 1:
                    raise ValueError(f"cat_reshape: {cls.__name__}: {old!r} found {source.count(old)} times")
                source = source.replace(old, new)
            if "super(" in source or "__class__" in source:
                raise ValueError(f"cat_reshape: {cls.__name__}.forward uses super(); cannot re-create it")
            namespace: dict[str, Any] = {}
            exec(compile(source, f"<cat_reshape:{cls.__name__}.forward>", "exec"), inspect.getmodule(cls).__dict__, namespace)
            cls.forward = namespace["forward"]
        return {"name": name, "classes": [c.__name__ for c in owners], "replaced": [o for o, _ in pairs]}
    if name == "ipg_cache":
        if loaded.method != "IPG":
            raise ValueError(f"{name} applies to IPG only")
        found = {}
        for meth, needle in (("calc_graph", "self.tensors=("),
                             ("rel_pos_bias", "self.relative_position_bias_table=relative_position_bias_table")):
            owners = {type(m) for m in loaded.model.modules() if meth in type(m).__dict__}
            if len(owners) != 1:
                raise ValueError(f"{name}: expected one class with {meth}, found {[c.__name__ for c in owners]}")
            cls = owners.pop()
            if needle not in "".join(inspect.getsource(cls.__dict__[meth]).split()):
                raise ValueError(f"{name}: released cache {needle!r} not found in {cls.__name__}.{meth}")
            found[f"{cls.__name__}.{meth}"] = sum(isinstance(m, cls) for m in loaded.model.modules())
        return {"name": name, "eager_first_forward": True, "caches": found}
    if name in ("ipg_const", "ipg_eager"):
        if loaded.method != "IPG":
            raise ValueError(f"{name} applies to IPG only")
        owners = {type(m) for m in loaded.model.modules() if "calc_graph" in type(m).__dict__}
        if len(owners) != 1:
            raise ValueError(f"{name}: expected one class with calc_graph, found {[c.__name__ for c in owners]}")
        cls = owners.pop()
        if name == "ipg_eager":
            cls.calc_graph = torch.compiler.disable(cls.__dict__["calc_graph"])
            return {"name": name, "class": cls.__name__, "method": "calc_graph"}
        compact = "".join(inspect.getsource(cls.__dict__["calc_graph"]).split())
        released = ("self.tensors=(torch.tensor([[0.5,1.,0.],[0.,0.,0.],[0.5,0.,1.],],dtype=torch.float32).to(x_.device),"
                    "torch.tensor([[0.5,0.,1.],[0.5,1.,0.],[0.,0.,0.],],dtype=torch.float32).to(x_.device))")
        if released not in compact:
            raise ValueError("ipg_const: the released constant tensors were not found in calc_graph")
        a = [[0.5, 1., 0.], [0., 0., 0.], [0.5, 0., 1.]]
        b = [[0.5, 0., 1.], [0.5, 1., 0.], [0., 0., 0.]]
        device = torch.device("cuda", torch.cuda.current_device())
        n = 0
        for m in loaded.model.modules():
            if isinstance(m, cls) and m.fast_graph and m.tensors is None:
                m.tensors = (torch.tensor(a, dtype=torch.float32, device=device),
                             torch.tensor(b, dtype=torch.float32, device=device))
                n += 1
        return {"name": name, "class": cls.__name__, "modules": n}
    raise ValueError(f"unknown fallback {name!r}")
