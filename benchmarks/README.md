# Super-resolution cost benchmark

This package produces every latency and peak-memory value of the paper (Figure 1, Tables 3, 4 and 13-16, Figure 8)
and the Params and FLOPs columns. All 21 methods run under one script and one software stack:

| Classical | Lightweight |
|---|---|
| SwinIR, CAT-A, ART, HAT, SRFormer, IPG, ATD, MambaIR, MambaIRv2, PFT, IET, Graphene | SwinIR-light, SRFormer-light, ATD-light, MambaIR-light, MambaIRv2-light, PFT-light, UCAN, IET-light, Graphene-light |

Every network comes from the authors' official repository, cloned at a pinned
commit (`third_party.json`). No model code is copied into this package.
`methods.py` contains one loader per method. It imports the architecture from
the clone, builds it from the official test configuration, and loads the
official checkpoint strictly after a sha256 check against `checkpoints.json`.

## Layout

```
benchmarks/
  methods.py            loaders, pinned windows, random-init parameter assertions, code variants (apply_fallback)
  counting.py           fvcore FLOP counting with handlers for the custom operators
  profile.py            CLI: run (worker / repeated runner), count, jobs, aggregate, env
  paper.py              job lists and value selection for the paper's tables (see Reproducing the tables)
  run_queue.py          serial queue for timing on an idle node (one fresh process per job)
  build_smm.py          builds smm_cuda of PFT-SR and of IET (two different sources)
  setup_third_party.sh  clone at pinned commits, build NATTEN and smm_cuda, list checkpoints
  third_party.json      repository URL + pinned commit per method
  checkpoints.json      checkpoint file, sha256 and official source per method and scale
  reported/             the reported value of every table entry with its runs (bf16-64, bf16-large, fp32-64, fp32-large)
  figures/              Figure 1 (BF16 compiled) and Figure 8 (FP32 eager): data and plotting scripts
  patches/              NATTEN 0.17.5 import fix for current torch
third_party/            clones of the baselines (created by setup_third_party.sh)
checkpoints/            official checkpoints, <Method>/<file> as listed in checkpoints.json (Graphene's may stay flat)
build/                  natten/ (pip --target) and smm/<repo>/ (extension builds)
```

## Setup

Tested environment: Python 3.12.3, torch 2.9.1+cu130, CUDA 13.0, cuDNN 9.23 and
an NVIDIA GH200 (SM 9.0). Python packages: `numpy`, `pyyaml`, `opencv-python`,
`timm`, `einops` and `fvcore`. MambaIR and
MambaIRv2 also need `mamba_ssm` and `causal_conv1d`. Building NATTEN and smm_cuda
needs the Python development headers (`python3-dev`) and `cmake` on `PATH`.

```bash
bash benchmarks/setup_third_party.sh clone        # official repos at pinned commits (+ NATTEN patch)
CUDA_HOME=/usr/local/cuda-13.0 CUDA_ARCH=9.0 BUILD_JOBS=32 \
  bash benchmarks/setup_third_party.sh natten     # NATTEN 0.17.5 -> build/natten
CUDA_HOME=/usr/local/cuda-13.0 CUDA_ARCH=9.0 \
  bash benchmarks/setup_third_party.sh smm        # smm_cuda for PFT-SR and IET -> build/smm/<repo>
bash benchmarks/setup_third_party.sh checkpoints  # lists every file, sha256 and official source
```

The script never downloads weights. Download the files listed by `checkpoints` from the official sources into
`checkpoints/<file>`. Graphene's code is this repository and its weights are the reference checkpoints in
`checkpoints/` (see the repository README); `--graphene-root` selects another copy of the Graphene code.

## Protocol

Two execution settings, each applied to every method:

* **BF16 compiled** (Figure 1, Tables 3, 4, 13, 14): BF16 autocast with FP32 weights, `torch.compile` with the Inductor
  backend in max-autotune mode (`dynamic=False`), TF32 off. Graphene keeps its released FP32 input/output convolutions
  under autocast (`model.fp32_io = True`).
* **FP32 eager** (Tables 15, 16, Figure 8): FP32, no compilation, TF32 off.

Common to both:

* TF32 is off for matmul and cuDNN, `cudnn.benchmark = True`, `torch.set_float32_matmul_precision('highest')`. Each
  process runs `torch.manual_seed(2447)` before building the model and works under `torch.inference_mode()`.
* The input is `torch.randn(1, 3, H, W)`: 64x64 for the main tables, and 180x320 and 720x1280 (H x W, i.e. 320x180 and
  1280x720 LR images) for Tables 4 and 16.
* The input is flip-padded at the right/bottom to a multiple of the method's window, and the output is cropped back to
  `scale*H x scale*W`. Window (padding multiple) per method: SwinIR(-light) 8, ART 8, SRFormer 22, IPG lcm(16, 32) = 32,
  PFT(-light) 32, IET(-light) 32, Graphene 32; 16 for all others. At 64x64 only SRFormer pads (to 66x66). IET
  additionally pads inside its forward to a multiple of 75.
* Each run is a fresh process: warm-up passes, `empty_cache()`, a peak-memory reset, then forward passes timed with CUDA
  events. The first timed pass re-allocates the activation buffers after the reset, so the latency of a run is the mean
  of the remaining passes. 64x64: 50 warm-up and 200 timed passes, latency = median over 7 runs. 180x320 and 720x1280:
  5 warm-up and 20 timed passes, median over 3 runs.
* Peak memory is `torch.cuda.max_memory_allocated()` (weights included). Max-autotune replays CUDA graphs, whose memory
  pool is invisible to the allocator statistics, so the peak memory of a max-autotune entry is the median of three runs
  of the same compilation without CUDA graphs (max-autotune-no-cudagraphs). Other entries report the memory of their
  timing runs.
* Params is the sum of `numel` over `model.parameters()`. FLOPs are 2 x MACs from fvcore at 1x3x64x64, using the handlers
  in `counting.py`. Operators without multiply-accumulate work are listed in the JSON and add nothing.

**Code variants (BF16 compiled only).** The released code of every method runs unchanged, except where one part cannot
run this setting; only that part is changed (`--fallback`, see `methods.apply_fallback`):

| Method | Reason | Variants measured |
|---|---|---|
| CAT-A | Inductor's lowering of `torch.roll` returns non-contiguous strides, so four `view` calls per shifted block fail to compile | `cat_reshape` (`view` -> `reshape`, same values), `cat_eager` (shifted blocks eager) |
| IPG | graph-construction constants and the position-bias table are cached at the first forward pass; CUDA-graph replays overwrite them | `ipg_cache` (one eager pass with the same autocast fills them), released code without CUDA graphs |
| PFT, PFT-light | the sparse-attention CUDA operators accept FP32 but not BF16 | `smm_fp32` (these calls in FP32, all else BF16); PFT-light also runs unchanged |
| IET, IET-light | the sparse-attention CUDA operators accept FP32 and FP16 but not BF16 | `smm_fp16`, `smm_fp32`, each with NATTEN's FP32 kernels or its GEMM kernels with TF32 off; IET-light also runs unchanged |

Every variant is measured, and the fastest is reported, the one most favourable to the baseline (`paper.py`). At
720x1280, matrix products with 2^31 or more elements in an operand or the output are lowered to cuBLAS, because the
Triton matrix-multiplication templates of Inductor index with 32-bit offsets (`install_triton_gemm_int32_guard` in
`profile.py`); where no variant of a method completes in max-autotune, the variants are run without CUDA graphs and in
the default mode. Out-of-memory runs are reported as OOM. In FP32 eager inference every method runs its released code
unchanged; IET and IET-light use NATTEN's FP32 kernels.

Every result JSON records the Python, torch, CUDA, cuDNN, GPU and driver versions, the CPU affinity, the commit (and
tracked-file state) of the repository used, the checkpoint sha256, the code variant, and the smm_cuda/NATTEN library
sha256 together with the NATTEN kernel flags.

## Commands

Run on an otherwise idle GPU and pin the process to the CPUs local to that GPU (CPUs 0-71 for GPU 0 on a GH200 node).

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72

# Params / FLOPs
taskset -c 0-71 python3 benchmarks/profile.py count --method HAT --scale 4 --output results/counts/HAT_x4.json

# one BF16 compiled run (latency) and one memory run of the same compilation without CUDA graphs
taskset -c 0-71 python3 benchmarks/profile.py run --method HAT --scale 4 --execution bf16_compiled \
    --compile-mode max-autotune --output results/HAT_x4_bf16_r1.json
taskset -c 0-71 python3 benchmarks/profile.py run --method HAT --scale 4 --execution bf16_compiled \
    --compile-mode max-autotune-no-cudagraphs --output results/HAT_x4_bf16_mem_r1.json

# one FP32 eager run at 1280x720
taskset -c 0-71 python3 benchmarks/profile.py run --method HAT --scale 4 --execution fp32_eager \
    --height 720 --width 1280 --warmup 5 --iterations 20 --output results/HAT_x4_fp32_720x1280_r1.json
```

## Reproducing the tables

`paper.py` writes the job lists of each table and selects the reported values. Tables: `bf16-64` (Figure 1, Tables 3,
13, 14), `bf16-large` (Table 4), `fp32-64` (Table 15, Figure 8), `fp32-large` (Table 16).

```bash
T=bf16-64
python3 benchmarks/paper.py jobs --table $T --stage probe --output results/$T/jobs_probe.json
python3 benchmarks/run_queue.py results/$T/jobs_probe.json --status results/$T/queue_probe.json
python3 benchmarks/paper.py jobs --table $T --stage full --skip-done --output results/$T/jobs_full.json  # repeat both lines
python3 benchmarks/run_queue.py results/$T/jobs_full.json --status results/$T/queue_full.json  # until it lists 0 jobs
python3 benchmarks/paper.py jobs --table $T --stage memory --output results/$T/jobs_memory.json
python3 benchmarks/run_queue.py results/$T/jobs_memory.json --status results/$T/queue_memory.json
python3 benchmarks/paper.py select --table $T --output results/$T/reported.json
```

The probe stage runs every variant once; the full stage first probes, for a method with no completing variant, the
other compilation modes, then repeats every variant within 1.25x of the method's fastest probe; the memory stage runs
the no-CUDA-graph memory runs of the picks. For the FP32 tables, `jobs --stage full` and `select` suffice.
`reported/` holds the result of `select` for the measurements in the paper: the reported variant per method, scale and
size with all its run latencies, the median latency and the reported peak memory. Its values print as Tables 3, 4 and
13-16 and as Figures 1 and 8 (`figures/`).

## Notes

* **Random initialisation.** SwinIR-light, SRFormer-light, ATD-light,
  MambaIR-light and MambaIRv2-light have no public checkpoint at the pinned
  commits. They are built after `torch.manual_seed(2447)` from the official
  configurations: SwinIR-light from the `lightweight_sr` arguments of
  `main_test_swinir.py`, the others from the official test yml. The loader
  asserts their exact parameter counts. Their cost does not depend on the
  weight values. All other methods, including PFT-light, use official weights.
* **NATTEN.** IET and IET-light call `natten.functional.na2d_qk/na2d_av` in the
  first block of each group. NATTEN 0.17.5 is built from source against the
  local torch, with CUDA and CUTLASS kernels for `NATTEN_CUDA_ARCH`. The patch
  in `patches/` only replaces an import that recent torch removed
  (`torch.cuda._device_t`). `--natten-mode` selects the kernels:
  `fp32` (default) turns NATTEN's GEMM kernels off (`natten.use_gemm_na(False)`)
  and gives true FP32 results, about 1e-6 from a float64 reference.
  `natten_default` keeps NATTEN's defaults (GEMM kernels with TF32, about 5e-3
  error on SM 9.0). `gemm_tf32_off` calls `natten.use_tf32_in_gemm_na(False)`,
  which on SM 9.0 still leaves TF32-level error. The default follows the
  protocol's TF32-off rule.
* **FLOPs of NATTEN.** For comparability, the table formula for NATTEN calls is
  `prod(out) * in[-1] / out[-1]`. It omits the kernel-area factor K^2 for QK and
  the head dimension D for AV. `count` also reports the full B*heads*H*W*K^2*D
  per call (`natten.flops_g_with_exact_natten`).
* **Counting-only changes.** ATD-light's official configuration enables
  `use_checkpoint`, so `torch.utils.checkpoint` wraps its layers in forward;
  this is kept for timing. For the fvcore trace it is switched off, which does
  not change the MACs. IET's `batched_isin` is traced with an int clamp bound
  (`counting.trace_safe_batched_isin`). Both are recorded in the count JSON.
* **smm_cuda.** PFT-SR and IET ship different `ops_smm` sources under the same
  module name. `build_smm.py` builds each with the official flags into its own
  directory, and each method imports its own build.
* **Import plumbing.** Repositories that carry a BasicSR copy are imported
  through that copy without running `basicsr/__init__.py`. HAT (which imports
  the pip `basicsr`) and UCAN get a minimal registry plus timm's
  `to_2tuple`/`trunc_normal_`. PFT and IET get a pass-through
  `fairscale.nn.checkpoint_wrapper`, which their code applies to no layer.
  See the docstring of `methods.py`.
