"""Full-image classical, lightweight, and real-world Graphene inference."""

import argparse
import json
from pathlib import Path

import torch

from models.build import build_model, read_config
from tools.images import flip_pad, image_files, read_rgb, to_tensor, to_uint8, write_rgb, psnr_ssim


@torch.inference_mode()
def reconstruct(model, image, config, device, precision=None):
    settings = config["inference"]
    precision = precision or settings["precision"]
    x = to_tensor(image).unsqueeze(0).to(device)
    height, width = image.shape[:2]
    if settings["padding"] == "flip":
        x = flip_pad(x, config["model"]["window_size"])
    elif settings["padding"] != "reflect":
        raise ValueError(f"Unknown padding: {settings['padding']}")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        output = model(x)
    scale = config["model"]["upscale"]
    return to_uint8(output[..., :height * scale, :width * scale])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Directory containing LR RGB images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hr", help="Optional corresponding HR directory for PSNR/SSIM")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    device = torch.device(args.device)
    tf32 = config["inference"].get("tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = config["inference"].get("cudnn_tf32", tf32)
    torch.backends.cudnn.benchmark = config["inference"].get("cudnn_benchmark", True)
    model = build_model(config, args.checkpoint).to(device).eval()
    if args.compile:
        model = torch.compile(model, dynamic=False)
    inputs = image_files(args.input)
    if not inputs:
        raise ValueError(f"No images in {args.input}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in inputs:
        relative = source.relative_to(args.input)
        prediction = reconstruct(model, read_rgb(source), config, device, args.precision)
        write_rgb(output / relative.with_suffix(".png"), prediction)
        row = {"image": str(relative)}
        if args.hr:
            from tools.train import find_target
            target = find_target(relative, Path(args.hr), config["model"]["upscale"])
            row["psnr"], row["ssim"] = psnr_ssim(prediction, read_rgb(target), config["model"]["upscale"])
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"model": config["name"], "precision": args.precision or config["inference"]["precision"], "images": results}
    if args.hr:
        for key in ("psnr", "ssim"):
            report[key] = sum(row[key] for row in results) / len(results)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "images"}), flush=True)


if __name__ == "__main__":
    main()
