#!/usr/bin/env python3
"""Cost benchmark: latency / peak memory worker and runner, Params/FLOPs counter, job lists.

Commands (the first positional argument, default ``run``):

  run        Measure one (method, scale, size). With --repeats 1 (default) the
             measurement runs in this process; this is the unit a job queue
             runs. With --repeats N the command starts N fresh worker processes
             one after another, writes <output stem>_r<i>.json for each, and
             writes the median summary to --output.
  count      Params (sum of numel) and FLOPs (fvcore, 2 x MACs) at 1x3x64x64.
  jobs       Write a job list (one fresh process per entry) for a queue runner.
  aggregate  Median over repeated runs found in the given files/directories.
  env        Print the software/hardware environment and the third-party state.

Protocol (fixed unless a flag says otherwise): FP32 eager, TF32 off for matmul
and cuDNN, cudnn.benchmark on, float32 matmul precision 'highest',
torch.manual_seed(2447), torch.inference_mode; input randn(1, 3, H, W); right/bottom
flip padding to the method's window multiple and cropping of the output; 50
warm-up and 200 CUDA-event-timed forward passes; latency = mean of timed passes 2..200 (pass 1 follows the allocator reset);
memory = torch.cuda.max_memory_allocated() after a reset that follows the
warm-up (weights included); median over fresh processes (seven at 64x64, three at
large inputs; set by the job lists).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Never write bytecode into the third-party clones (some track __pycache__ files).
sys.dont_write_bytecode = True

if __package__ in (None, ""):
    # Run as a script: replace the script directory on sys.path by the package root, so this file
    # never shadows the standard-library `profile` module (imported by cProfile inside torch).
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    __package__ = "benchmarks"

from benchmarks import methods  # noqa: E402

PROTOCOL = {"warmup": 50, "iterations": 200, "repeats": 3, "seed": 2447, "height": 64, "width": 64}
EXECUTIONS = ("fp32_eager", "bf16_compiled", "bf16_eager", "fp32_compiled", "fp16_eager", "fp16_compiled")
TF32_SETTINGS = ("off", "torch_default", "on")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")
TABLE_SIZES = ("64x64", "180x320", "720x1280")  # HxW: 64x64, 320x180 and 1280x720 inputs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "count", "jobs", "aggregate", "env"])
    p.add_argument("paths", nargs="*", type=Path, help="aggregate: result files or directories")
    p.add_argument("--method", choices=methods.METHODS)
    p.add_argument("--scale", type=int, choices=methods.SCALES, default=4)
    p.add_argument("--execution", choices=EXECUTIONS, help="queue name of precision+compile (default fp32_eager)")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], help="fp32, or bf16 / fp16 autocast")
    p.add_argument("--compile", action="store_true", help="torch.compile(backend='inductor', dynamic=False)")
    p.add_argument("--compile-mode", "--compile_mode", dest="compile_mode", choices=COMPILE_MODES, default="default",
                   help="torch.compile mode for *_compiled executions (default: inductor's default mode)")
    p.add_argument("--cudnn-benchmark", "--cudnn_benchmark", dest="cudnn_benchmark", choices=["on", "off"], default="on",
                   help="cudnn.benchmark autotuning (protocol: on)")
    p.add_argument("--tf32", choices=TF32_SETTINGS, default="off",
                   help="off = TF32 disabled for matmul and cuDNN, matmul precision 'highest' (protocol); "
                        "torch_default = leave PyTorch's own defaults untouched (as a released inference script does); "
                        "on = TF32 enabled for matmul and cuDNN, matmul precision 'high'")
    p.add_argument("--height", type=int, default=PROTOCOL["height"])
    p.add_argument("--width", type=int, default=PROTOCOL["width"])
    p.add_argument("--warmup", type=int, default=PROTOCOL["warmup"])
    p.add_argument("--iterations", type=int, default=PROTOCOL["iterations"])
    p.add_argument("--repeats", type=int, default=1, help="fresh processes; >1 makes `run` a runner")
    p.add_argument("--seed", type=int, default=PROTOCOL["seed"])
    p.add_argument("--threads", type=int, default=72, help="torch.set_num_threads (pin the process with taskset)")
    p.add_argument("--natten-mode", "--natten_mode", dest="natten_mode", choices=methods.NATTEN_MODES, default="fp32",
                   help="IET only: fp32 = NATTEN's non-GEMM FP32 kernels (default, matches the TF32-off protocol); "
                        "natten_default = NATTEN's defaults (GEMM kernels with TF32); "
                        "gemm_tf32_off = GEMM kernels with natten.use_tf32_in_gemm_na(False)")
    p.add_argument("--fallback", choices=methods.FALLBACKS, default="none",
                   help="change only the part of a released model that cannot run BF16 compiled inference "
                        "(see methods.apply_fallback); none = released code unchanged")
    p.add_argument("--third-party", type=Path)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--build", type=Path)
    p.add_argument("--graphene-root", type=Path)
    p.add_argument("--no-verify-checkpoints", action="store_true", help="skip the sha256 check against checkpoints.json")
    p.add_argument("--allow-multiple-gpus", action="store_true")
    p.add_argument("--memory-only", action="store_true",
                   help="run the same passes without CUDA-event timing (peak memory and code-path check only)")
    p.add_argument("--output", type=Path)
    # jobs
    p.add_argument("--methods", nargs="+", choices=methods.METHODS, default=methods.METHODS)
    p.add_argument("--scales", nargs="+", type=int, choices=methods.SCALES, default=list(methods.SCALES))
    p.add_argument("--sizes", nargs="+", default=["64x64"], help="HxW, e.g. 64x64 180x320 720x1280")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    args = p.parse_args(argv)
    args.execution = resolve_execution(args)
    return args


def resolve_execution(args) -> str:
    if args.execution and (args.precision or args.compile):
        derived = f"{args.precision or 'fp32'}_{'compiled' if args.compile else 'eager'}"
        if derived != args.execution:
            raise SystemExit(f"--execution {args.execution} conflicts with --precision/--compile ({derived})")
    if args.execution:
        return args.execution
    return f"{args.precision or 'fp32'}_{'compiled' if args.compile else 'eager'}"


def roots_from(args) -> methods.Roots:
    return methods.Roots.from_args(args.third_party, args.checkpoints, args.build, args.graphene_root)


# --------------------------------------------------------------------------- environment

def _version(name: str):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"not importable ({type(exc).__name__})"


def environment(device_index: int = 0) -> dict:
    import torch

    env = {
        "python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "packages": {name: _version(name) for name in ("numpy", "fvcore", "timm", "einops", "yaml", "mamba_ssm", "triton")},
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")},
    }
    try:
        cpus = sorted(os.sched_getaffinity(0))
        env["cpu_affinity"] = f"{len(cpus)} cpus: {cpus[0]}-{cpus[-1]}" if cpus else "none"
    except AttributeError:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        env.update(gpu=props.name, gpu_capability=f"{props.major}.{props.minor}",
                   gpu_memory_gib=round(props.total_memory / 2**30, 1), visible_gpus=torch.cuda.device_count())
        try:
            uuid = str(props.uuid)
            uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            out = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=driver_version,pci.bus_id",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            driver, bus = [s.strip() for s in out.stdout.strip().split(",")]
            env.update(driver=driver, gpu_pci_bus_id=bus)
        except Exception:
            pass
    return env


def configure_torch(threads: int, seed: int, tf32: str = "off", cudnn_benchmark: bool = True):
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    elif tf32 == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return torch


def tf32_state(torch) -> dict:
    return {"matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def loaded_record(loaded: methods.Loaded) -> dict:
    return {"weights": loaded.weights, "window": loaded.window, "repo": loaded.repo, "config": loaded.config,
            "checkpoint": loaded.checkpoint, "extensions": loaded.extensions, "notes": loaded.notes}


# --------------------------------------------------------------------------- run (one process)

def install_triton_gemm_int32_guard() -> dict:
    """Inductor (PyTorch 2.9.1) Triton GEMM templates index operands and output with 32-bit offsets. When an operand
    or the output of a matmul has at least 2**31 elements, every template config raises a CUDA illegal memory access
    during max-autotune, which also poisons the process. At 1280x720 LR this happens for HAT (bmm 21600x256x32 @
    21600x32x576 -> 3.19e9-element output, then that tensor @ 21600x576x32) and CAT-A (1920x1440x1440).
    Standalone reproduction: 2,147,401,728 elements run, 2,147,549,184 fail.
    v2 (11-17 03:2x): only for such matmuls (aten.mm/addmm/bmm/baddbmm lowerings) is the GEMM backend list restricted
    to ATen; every other matmul keeps the full max-autotune choice set, so smaller GEMMs are unaffected.
    v1 checked only the output size and missed the attention-times-value product. The guarded matmul shapes are
    recorded in the result."""
    import torch
    from torch._inductor import config as icfg
    from torch._inductor import lowering as L
    aten = torch.ops.aten
    skipped = []

    def size(n):
        try:
            return [int(d) for d in n.get_size()]
        except Exception:
            return None

    def prod(s):
        p = 1
        for d in s:
            p *= d
        return p

    packets = {aten.mm, aten.addmm, aten.bmm, aten.baddbmm}
    for op, fn in list(L.lowerings.items()):
        if getattr(op, "overloadpacket", None) not in packets or getattr(fn, "_int32_guard", False):
            continue

        def guarded(*args, _fn=fn, _name=str(op), **kw):
            ts = [size(a) for a in args if hasattr(a, "get_size")]
            ts = [s for s in ts if s]
            big = False
            if len(ts) >= 2:
                s1, s2 = ts[-2], ts[-1]
                out = prod(s1[:-1]) * s2[-1]
                big = max([prod(s) for s in ts] + [out]) >= 2**31
            if not big:
                return _fn(*args, **kw)
            skipped.append([_name] + ts)
            with icfg.patch(max_autotune_gemm_backends="ATEN"):
                return _fn(*args, **kw)
        guarded._int32_guard = True
        L.lowerings[op] = guarded
    # 11-17 04:4x: Inductor's FX-graph and AOTAutograd caches key on the graph and config, not on this patch, so a graph
    # cached before guard v2 was reused without lowering (no guard). A no-op post-grad pass whose uuid enters both cache
    # keys separates v2-lowered graphs from older entries; the autotune timing cache is unaffected.
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class _Int32GuardTag(CustomGraphPass):
        def __call__(self, graph):
            return None

        def uuid(self):
            return "triton_gemm_int32_guard_v2"
    tag = None
    if icfg.post_grad_custom_post_pass is None:
        icfg.post_grad_custom_post_pass = _Int32GuardTag()
        tag = "triton_gemm_int32_guard_v2"
    return {"version": 2, "threshold_elements": 2**31, "rule": "operand or output >= 2**31 elements -> ATen only",
            "guarded_matmuls": skipped, "cache_key_tag": tag}


def measure(args) -> int:
    torch = configure_torch(args.threads, args.seed, args.tf32, args.cudnn_benchmark == "on")
    if not args.allow_multiple_gpus and torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()} "
                         "(set CUDA_VISIBLE_DEVICES)")
    device = torch.device("cuda:0")
    bf16 = args.execution.startswith("bf16")
    fp16 = args.execution.startswith("fp16")  # for methods whose CUDA operators accept FP16 but not BF16
    compiled = args.execution.endswith("compiled")
    report = {
        "method": args.method, "scale": args.scale, "execution": args.execution,
        "input_shape": [1, 3, args.height, args.width], "warmup": args.warmup, "iterations": args.iterations,
        "seed": args.seed, "cpu_threads": args.threads, "natten_mode": args.natten_mode, "tf32": args.tf32,
        "compile_mode": args.compile_mode if compiled else None,
        "protocol": {"tf32": tf32_state(torch) if args.tf32 != "off" else False, "cudnn_benchmark": args.cudnn_benchmark == "on",
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "autocast_bf16": bf16,
                     "autocast_fp16": fp16,
                     "torch_compile": f"inductor, mode={args.compile_mode}, dynamic=False" if compiled else None,
                     "padding": "right/bottom flip to window multiple, output cropped",
                     "latency": "mean of CUDA-event times of timed passes 2..N (pass 1 re-allocates after the allocator reset)",
                     "memory": "max_memory_allocated after reset following warm-up (weights included)"},
        "protocol_conforming": (args.warmup >= PROTOCOL["warmup"] and args.iterations >= PROTOCOL["iterations"]
                                and not args.memory_only and args.tf32 == "off" and args.cudnn_benchmark == "on"),
        "memory_only": args.memory_only,
        "environment": environment(), "started": time.time(), "status": "running",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        loaded = methods.load(args.method, args.scale, roots_from(args), args.natten_mode,
                              not args.no_verify_checkpoints)
        report.update(loaded_record(loaded), parameters=sum(p.numel() for p in loaded.model.parameters()))
        if args.fallback != "none":
            report["fallback"] = methods.apply_fallback(loaded, args.fallback)
        model = loaded.model.to(device).eval()
        window = loaded.window
        h_pad = math.ceil(args.height / window) * window
        w_pad = math.ceil(args.width / window) * window
        report["padded_input_shape"] = [1, 3, h_pad, w_pad]
        if loaded.prepare is not None:
            loaded.prepare(model, h_pad, w_pad)

        def pad(x):
            h, w = x.shape[-2:]
            ph, pw = (window - h % window) % window, (window - w % window) % window
            if ph:
                x = torch.cat([x, torch.flip(x, [-2])], -2)[..., : h + ph, :]
            if pw:
                x = torch.cat([x, torch.flip(x, [-1])], -1)[..., : w + pw]
            return x

        if compiled:
            # 11-17: see install_triton_gemm_int32_guard (v2); no effect unless a matmul operand or output has >= 2**31 elements
            report["triton_gemm_int32_guard"] = install_triton_gemm_int32_guard()
        net = torch.compile(model, backend="inductor", mode=args.compile_mode, dynamic=False) if compiled else model
        cudagraphs = compiled and args.compile_mode in CUDAGRAPH_MODES

        def forward(x):
            h, w = x.shape[-2:]
            if cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            return net(pad(x))[..., : h * args.scale, : w * args.scale]

        x = torch.randn(1, 3, args.height, args.width, device=device)
        if report.get("fallback", {}).get("eager_first_forward"):
            # methods.apply_fallback (ipg_cache): one eager pass fills the released first-forward caches before the
            # first compiled call; it precedes warm-up and the allocator reset, so it is not timed or counted
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16 if fp16 else torch.bfloat16,
                                                        enabled=bf16 or fp16):
                model(pad(x))
            torch.cuda.synchronize()
            report["fallback"]["eager_first_forward_done"] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16 if fp16 else torch.bfloat16, enabled=bf16 or fp16):
            for i in range(args.warmup):
                y = forward(x)
                if i == 0:
                    # Peak over warm-up passes 2..N: excludes compilation/autotuning in the first pass but includes
                    # CUDA graph capture, whose private-pool memory max_memory_allocated() no longer sees at replay.
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            warmup_peak = torch.cuda.max_memory_allocated() if args.warmup > 1 else None
            del y
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if args.memory_only:
                for _ in range(args.iterations):
                    y = forward(x)
            else:
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                for start, end in zip(starts, ends):
                    start.record()
                    y = forward(x)
                    end.record()
            torch.cuda.synchronize()
        if not args.memory_only:
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            # The first timed pass follows empty_cache() and re-allocates every activation buffer (up to 14x a steady
            # pass at 720x1280); latency statistics use passes 2..N, times_ms keeps every pass.
            timed = times[1:] if len(times) > 1 else times
            report.update(latency_ms=statistics.mean(timed), latency_median_ms=statistics.median(timed),
                          latency_stdev_ms=statistics.pstdev(timed), times_ms=times, latency_excludes_first_pass=len(times) > 1)
        report.update(
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20, baseline_memory_mib=baseline / 2**20,
            warmup_peak_memory_mib=warmup_peak / 2**20 if warmup_peak is not None else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, cudagraphs=cudagraphs,
            output_shape=list(y.shape), output_finite=bool(torch.isfinite(y).all()), status="complete")
        if compiled:
            from torch._dynamo.utils import counters
            report["compiler_counters"] = {k: dict(v) for k, v in counters.items()}
    except Exception as exc:
        report.update(status="oom" if type(exc).__name__ == "OutOfMemoryError" else "failed",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished"] = time.time()
        save()
        print(json.dumps({k: v for k, v in report.items()
                          if k in ("method", "scale", "execution", "input_shape", "status", "latency_ms",
                                   "peak_memory_mib", "parameters", "weights", "error")}), flush=True)
    return 0 if report["status"] == "complete" else 1


# --------------------------------------------------------------------------- run (runner)

def worker_argv(args, output: Path) -> list[str]:
    argv = [sys.executable, str(Path(__file__).resolve()), "run", "--method", args.method, "--scale", str(args.scale),
            "--execution", args.execution, "--height", str(args.height), "--width", str(args.width),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--seed", str(args.seed),
            "--threads", str(args.threads), "--natten-mode", args.natten_mode, "--tf32", args.tf32,
            "--compile-mode", args.compile_mode, "--fallback", args.fallback, "--output", str(output)]
    for flag, value in (("--third-party", args.third_party), ("--checkpoints", args.checkpoints),
                        ("--build", args.build), ("--graphene-root", args.graphene_root)):
        if value is not None:
            argv += [flag, str(value)]
    if args.no_verify_checkpoints:
        argv.append("--no-verify-checkpoints")
    if args.memory_only:
        argv.append("--memory-only")
    return argv


def run_repeats(args) -> int:
    if args.output is None:
        raise SystemExit("--output is required with --repeats > 1")
    outputs = [args.output.with_name(f"{args.output.stem}_r{i}.json") for i in range(1, args.repeats + 1)]
    codes = []
    for output in outputs:
        codes.append(subprocess.run(worker_argv(args, output), timeout=args.timeout).returncode)
    summary = summarize([json.loads(o.read_text()) for o in outputs if o.exists()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary[0].items() if k != "runs"} if summary else {}), flush=True)
    return 0 if all(c == 0 for c in codes) else 1


def summarize(reports: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in reports:
        key = (r.get("method"), r.get("scale"), r.get("execution"), tuple(r.get("input_shape") or ()),
               r.get("natten_mode") if str(r.get("method", "")).startswith("IET") else None,
               r.get("tf32") or "off", r.get("compile_mode") or "default")
        groups.setdefault(key, []).append(r)
    rows = []
    for (method, scale, execution, shape, natten_mode, tf32, compile_mode), runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        complete = [r for r in runs if r.get("status") == "complete"]
        row = {"method": method, "scale": scale, "execution": execution, "input_shape": list(shape),
               "runs_total": len(runs), "runs_complete": len(complete),
               "statuses": sorted({r.get("status") for r in runs})}
        if natten_mode:
            row["natten_mode"] = natten_mode
        if tf32 != "off":
            row["tf32"] = tf32
        if compile_mode != "default":
            row["compile_mode"] = compile_mode
        if complete:
            lat = [r["latency_ms"] for r in complete if r.get("latency_ms") is not None]
            mem = [r["peak_memory_mib"] for r in complete]
            row.update(latency_ms_median=statistics.median(lat) if lat else None, latency_ms_runs=lat,
                       peak_memory_mib_median=statistics.median(mem), peak_memory_mib_runs=mem,
                       parameters=complete[0].get("parameters"), weights=complete[0].get("weights"),
                       protocol_conforming=all(r.get("protocol_conforming") for r in complete)
                       and len(complete) >= PROTOCOL["repeats"])
        rows.append(row)
    return rows


def aggregate(args) -> int:
    files: list[Path] = []
    for path in args.paths:
        files += sorted(Path(p) for p in glob.glob(str(path / "**/*.json"), recursive=True)) if path.is_dir() else [path]
    reports = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and data.get("method") and data.get("status") in ("complete", "failed", "oom"):
            data["_file"] = str(f)
            reports.append(data)
    rows = summarize(reports)
    text = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


# --------------------------------------------------------------------------- count / jobs / env

def count_command(args) -> int:
    torch = configure_torch(args.threads, args.seed)
    from benchmarks import counting

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    report = {"method": args.method, "scale": args.scale, "natten_mode": args.natten_mode,
              "environment": environment() if device.type == "cuda" else None, "status": "running"}
    try:
        loaded = methods.load(args.method, args.scale, roots_from(args), args.natten_mode,
                              not args.no_verify_checkpoints)
        report.update(loaded_record(loaded))
        model = loaded.model.to(device).eval()
        if loaded.prepare is not None:
            loaded.prepare(model, args.height, args.width)
        report["count_only_patches"] = counting.tracing_patches(args.method)
        report.update(counting.count(model, (args.height, args.width), device))
    except Exception as exc:
        report.update(status="count_error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("method", "scale", "status", "params", "flops_g", "unknown_ops", "error")}))
    return 0 if report["status"] == "ok" else 1


def jobs_command(args) -> int:
    jobs = []
    for repeat in range(1, args.repeats + 1):  # all r1 first, then r2, ...: drift spreads over methods
        for size in args.sizes:
            height, width = (int(v) for v in size.lower().split("x"))
            for method in args.methods:
                for scale in args.scales:
                    natten = method.startswith("IET") and args.natten_mode != "fp32"
                    tag = f"_{args.natten_mode}" if natten else ""
                    name = f"{method}_x{scale}_{args.execution}_{height}x{width}{tag}_r{repeat}.json"
                    job = {"method": method, "scale": scale, "execution": args.execution,
                           "height": height, "width": width, "warmup": args.warmup,
                           "iterations": args.iterations, "timeout": args.timeout,
                           "output": str(args.output_dir / name)}
                    if natten:
                        job["natten_mode"] = args.natten_mode
                    jobs.append(job)
    text = json.dumps(jobs, indent=1) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(f"{len(jobs)} jobs" + (f" -> {args.output}" if args.output else ""))
    return 0


def env_command(args) -> int:
    roots = roots_from(args)
    report = {"environment": environment(),
              "repos": {name: methods.repo_state(roots, name) for name in methods.third_party_spec()}}
    builds = {}
    for name in ("PFT-SR", "IET"):
        d = roots.build / "smm" / name
        libs = sorted(d.glob("smm_cuda*.so"))
        builds[f"smm_cuda[{name}]"] = {"path": str(libs[0]), "sha256": methods.sha256_file(libs[0])} if libs else None
    lib = sorted((roots.build / "natten" / "natten").glob("libnatten*.so"))
    builds["natten"] = {"path": str(lib[0]), "sha256": methods.sha256_file(lib[0])} if lib else None
    report["builds"] = builds
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command in ("run", "count") and not args.method:
        raise SystemExit(f"{args.command}: --method is required")
    if args.command == "run":
        return run_repeats(args) if args.repeats > 1 else measure(args)
    if args.command == "count":
        return count_command(args)
    if args.command == "jobs":
        return jobs_command(args)
    if args.command == "aggregate":
        return aggregate(args)
    return env_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
