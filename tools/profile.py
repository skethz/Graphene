"""Graphene operation counts and fresh-process BF16 compiled runtime profiling."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import torch

from models.build import build_model, prepare_masks, read_config


NON_MAC_OPERATIONS = {
    "aten::add", "aten::add_", "aten::abs", "aten::cat", "aten::clone",
    "aten::clamp_min", "aten::diagonal", "aten::div", "aten::div_", "aten::eq",
    "aten::exp", "aten::fill_", "aten::flip", "aten::gelu", "aten::index_select",
    "aten::leaky_relu", "aten::leaky_relu_", "aten::lt", "aten::masked_fill",
    "aten::mean", "aten::meshgrid", "aten::mul", "aten::mul_", "aten::neg",
    "aten::pad", "aten::pixel_shuffle", "aten::repeat", "aten::roll", "aten::rsub",
    "aten::sigmoid", "aten::softmax", "aten::sub", "aten::sum", "aten::tanh",
    "aten::zero_", "aten::zeros_like", "aten::ones_like",
}


def operation_counts(model, x):
    from fvcore.nn import FlopCountAnalysis
    counter = FlopCountAnalysis(model, x)
    counter.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    macs = float(counter.total())
    unsupported = dict(counter.unsupported_ops())
    unknown = set(unsupported) - NON_MAC_OPERATIONS
    if unknown:
        raise RuntimeError(f"Uncounted operations need explicit handling: {sorted(unknown)}")
    return dict(parameters=sum(p.numel() for p in model.parameters()), macs=macs, flops=2 * macs,
                excluded_non_mac_operations=sorted(unsupported))


def worker(args):
    config = read_config(args.config)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    model = build_model(config, args.checkpoint).cuda().eval()
    prepare_masks(model, 64, 64)
    x = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float32)
    if args.counts_only:
        with torch.inference_mode():
            result = operation_counts(model, x)
    else:
        model.fp32_io = True
        model = torch.compile(model, backend="inductor", mode="default", dynamic=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.warmup):
                y = model(x)
            torch.cuda.synchronize()
            del y
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            for start, end in zip(starts, ends):
                start.record()
                y = model(x)
                end.record()
            torch.cuda.synchronize()
            elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            peak = torch.cuda.max_memory_allocated()
            result = dict(latency_ms=statistics.mean(elapsed), peak_memory_mib=peak / 2**20,
                          output_finite=bool(torch.isfinite(y).all()))
    result.update(model=config["name"], torch=torch.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                  input_shape=list(x.shape))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.runs, args.warmup, args.iterations, args.cpu_threads) < 1:
        parser.error("Run, pass, and thread counts must be positive.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.counts_only or args.worker:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="graphene_profile_") as directory:
        reports = []
        for index in range(args.runs):
            output = Path(directory) / f"run_{index}.json"
            subprocess.run([sys.executable, "-m", "tools.profile", "--worker", "--config", args.config,
                            "--checkpoint", args.checkpoint, "--output", str(output), "--warmup", str(args.warmup),
                            "--iterations", str(args.iterations), "--cpu-threads", str(args.cpu_threads)], check=True)
            reports.append(json.loads(output.read_text()))
        summary = dict(latency_ms=statistics.median(r["latency_ms"] for r in reports),
                       peak_memory_mib=statistics.median(r["peak_memory_mib"] for r in reports),
                       runs=reports, warmup=args.warmup, iterations=args.iterations)
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
