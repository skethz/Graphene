# Inference and evaluation

Select matching configuration and checkpoint files. All learned tensors must load successfully. Deterministic shifted-window masks are generated for the configured input size; no missing learned parameters are accepted.

Graphene-classical and Graphene-lightweight evaluation uses FP32, EMA weights, and one full-image forward pass. LR RGB values are divided by 255. Bottom/right flip concatenation completes each attention window, and predictions are cropped to the scaled original dimensions. Outputs are clamped to [0,1], multiplied by 255, rounded, and saved as 8-bit PNGs.

The Python evaluator matches the manuscript's MATLAB convention: `single(rgb2ycbcr(im2double(image)))`, retain Y, multiply by 255, and convert to double for metrics. It shaves `scale` pixels from all four sides. PSNR uses dynamic range 255. SSIM uses the 11×11 Gaussian window with σ=1.5, K₁=0.01 and K₂=0.03. Dataset averages are computed from per-image scores. Aligned HR images must match the output dimensions exactly.

The `graphene-realworld_x4.pth` checkpoint uses a nearest-neighbor/convolution ×4 reconstruction head, BF16 eager inference, and bottom/right reflection padding. The qualitative reference environment uses PyTorch 2.10.0, CUDA 13.0, and cuDNN 9.15.1 (`nvidia-cudnn-cu13==9.15.1.9`). Use `requirements-real.txt` in a separate environment. cuDNN benchmarking is disabled; cuDNN TF32 is enabled and matrix-multiplication TF32 is disabled. Real-world images have no aligned HR reference and are evaluated visually. A different kernel-library version can change BF16 output rounding.

## Efficiency

`tools.profile --counts-only` uses fvcore's operation counter at 1×3×64×64, with two FLOPs per MAC. The counter includes the encoder, IRB, HistAttn, and the reconstruction head. Listed elementwise/non-MAC operations are excluded consistently; unknown unsupported operations cause an error. Parameter counts include all trainable parameters, including the scale-specific head.

Runtime measurements use FP32 eager execution by default, with TF32 disabled. Add `--precision bf16 --compile` for an outer BF16 autocast context and `torch.compile(backend="inductor", mode="default", dynamic=False)`. Parameters and the released input/output path remain FP32. Deterministic attention masks are cached for the fixed input. Each run is a separate process with 50 warm-up and 200 CUDA-event-timed passes. Compilation is excluded from timing. Peak memory is CUDA allocated memory, including parameters, inputs, buffers, and activations; it is not reserved or nvidia-smi process memory.

The FP32 comparison retains the original baseline measurements and replaces only Graphene and Graphene-light with FP32 measurements. The BF16 comparison uses the Graphene reference points (27.7 ms / 238.2 MiB and 11.2 ms / 58.1 MiB at ×4) and measures the baselines with BF16 autocast and compilation. The manuscript appendix states the source and execution settings of the separate BF16 comparison.

Use an otherwise idle GPU and bind the process to its local NUMA CPU node where applicable. The profiling reference environment is Python 3.12.3, PyTorch 2.9.1, CUDA 13.0, cuDNN 9.23, and an NVIDIA GH200. Different kernel libraries, compiler versions, and device load can change latency, memory, and floating-point rounding. The command records the active PyTorch, CUDA, and cuDNN versions.
