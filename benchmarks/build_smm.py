#!/usr/bin/env python3
"""Build the smm_cuda extension of PFT-SR or IET into <build>/smm/<repo>.

The two repositories ship different ops_smm/src sources under the same module
name, so each gets its own build directory. Each method imports its own copy
(see methods._smm_cuda). The sources, include path, macro and nvcc flags are
the ones of the official ops_smm/setup.py. The only difference is that the
GPU architecture is given explicitly (--arch, default 9.0) instead of being
detected from a visible GPU, so the build needs no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", choices=["PFT-SR", "IET"], required=True)
    parser.add_argument("--third-party", type=Path, default=package_root / "third_party")
    parser.add_argument("--build", type=Path, default=package_root / "build")
    parser.add_argument("--arch", default="9.0", help="compute capability, e.g. 9.0 (GH200/H100), 8.0 (A100)")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    source_dir = (args.third_party / args.repo / "ops_smm" / "src").resolve()
    out_dir = (args.build / "smm" / args.repo).resolve()
    temp_dir = (args.build / "smm" / f"{args.repo}.tmp").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", str(args.jobs))

    import torch
    from setuptools import setup
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is None:
        raise SystemExit("CUDA_HOME not found")
    sources = sorted(str(p) for p in list(source_dir.glob("*.cpp")) + list(source_dir.glob("*.cu")))
    arch = args.arch.replace(".", "")
    nvcc = ["-gencode", f"arch=compute_{arch},code=sm_{arch}", "-O3", "-Xcompiler", "-Wall", "-lineinfo"]
    extension = CUDAExtension(name="smm_cuda", sources=sources, include_dirs=[str(source_dir)],
                              define_macros=[("WITH_CUDA", None)],
                              extra_compile_args={"cxx": [], "nvcc": nvcc})
    started = time.time()
    setup(name="smm_cuda", version="1.0", ext_modules=[extension], cmdclass={"build_ext": BuildExtension},
          script_args=["build_ext", "--build-lib", str(out_dir), "--build-temp", str(temp_dir)])
    shutil.rmtree(temp_dir, ignore_errors=True)
    libraries = sorted(out_dir.glob("smm_cuda*.so"))
    if not libraries:
        raise SystemExit("build produced no smm_cuda*.so")
    info = {
        "repo": args.repo, "sources": {Path(s).name: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        "nvcc_flags": nvcc, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": CUDA_HOME,
        "python": sys.version.split()[0], "library": libraries[0].name,
        "library_sha256": hashlib.sha256(libraries[0].read_bytes()).hexdigest(),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
