#!/usr/bin/env bash
# Clone the official repositories at their pinned commits, build NATTEN and the two smm_cuda
# extensions, and list the checkpoints to download. Run from the package root:
#
#   bash benchmarks/setup_third_party.sh [clone|natten|smm|checkpoints|all]   (default: all)
#
# Environment (defaults in brackets):
#   PYTHON            interpreter that runs the benchmark [python3]; NATTEN and smm_cuda are built for it
#   CUDA_HOME         CUDA toolkit matching torch.version.cuda [/usr/local/cuda]
#   CUDA_ARCH         GPU compute capability to build for [9.0]
#   BUILD_JOBS        parallel compile jobs [32]
#   CPATH             must contain Python.h and pyconfig.h if the python3-dev headers are not installed
#
# Weights are never downloaded by this script: `checkpoints` prints each file, its sha256 and its
# official source; place the files under checkpoints/<file>.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCH="${CUDA_ARCH:-9.0}"
BUILD_JOBS="${BUILD_JOBS:-32}"
TP="$ROOT/third_party"
BUILD="$ROOT/build"
STEP="${1:-all}"

repos() {  # name url commit [sparse path]
  "$PYTHON" - "$ROOT/benchmarks/third_party.json" <<'PY'
import json, sys
for name, spec in json.load(open(sys.argv[1]))["repos"].items():
    print(name, spec["url"], spec["commit"], " ".join(spec.get("sparse_checkout", [])))
PY
}

clone() {
  mkdir -p "$TP"
  repos | while read -r name url commit sparse; do
    dir="$TP/$name"
    if [ ! -d "$dir/.git" ]; then
      if [ -n "$sparse" ]; then
        git clone --filter=blob:none --no-checkout "$url" "$dir"
        git -C "$dir" sparse-checkout set $sparse
      else
        git clone "$url" "$dir"
      fi
    fi
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
    echo "$name $(git -C "$dir" rev-parse HEAD)"
  done
  git -C "$TP/NATTEN" submodule update --init --depth 1 third_party/cutlass
  if git -C "$TP/NATTEN" apply --check "$ROOT/benchmarks/patches/natten-0.17.5-torch-device-type.patch" 2>/dev/null; then
    git -C "$TP/NATTEN" apply "$ROOT/benchmarks/patches/natten-0.17.5-torch-device-type.patch"
  fi
  git -C "$TP/NATTEN" diff --stat
}

natten() {
  # NATTEN 0.17.5 from source, CUDA + CUTLASS kernels for CUDA_ARCH, installed only into build/natten.
  mkdir -p "$BUILD/natten" "$BUILD/tmp"
  (cd "$TP/NATTEN" && \
   CUDA_HOME="$CUDA_HOME" CUDACXX="$CUDA_HOME/bin/nvcc" PATH="$CUDA_HOME/bin:$PATH" \
   NATTEN_WITH_CUDA=1 NATTEN_CUDA_ARCH="$CUDA_ARCH" NATTEN_N_WORKERS="$BUILD_JOBS" \
   TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" CUDA_VISIBLE_DEVICES="" TMPDIR="$BUILD/tmp" \
   "$PYTHON" -m pip install --no-deps --no-build-isolation --no-cache-dir --target "$BUILD/natten" -v .)
}

smm() {
  for repo in PFT-SR IET; do
    CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" CUDA_VISIBLE_DEVICES="" \
      "$PYTHON" "$ROOT/benchmarks/build_smm.py" --repo "$repo" --arch "$CUDA_ARCH" --jobs "$BUILD_JOBS"
  done
}

checkpoints() {
  "$PYTHON" - "$ROOT/benchmarks/checkpoints.json" "$ROOT/checkpoints" <<'PY'
import hashlib, json, sys
from pathlib import Path
spec = json.load(open(sys.argv[1])); root = Path(sys.argv[2])
missing = 0
for method, scales in spec["checkpoints"].items():
    for scale, entry in scales.items():
        path = root / entry["file"]
        state = "missing"
        if path.is_file():
            state = "ok" if hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"] else "SHA256 MISMATCH"
        missing += state != "ok"
        print(f"{state:16s} {entry['file']}\n                 sha256 {entry['sha256']}\n                 source {entry['source']}")
print("random init (no checkpoint):", ", ".join(spec["random_init"]))
print(f"{missing} file(s) missing or mismatched")
PY
}

case "$STEP" in
  clone) clone ;;
  natten) natten ;;
  smm) smm ;;
  checkpoints) checkpoints ;;
  all) clone; natten; smm; checkpoints ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
