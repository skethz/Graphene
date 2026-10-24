"""RGB image conversion, window padding, and Y-channel SR metrics."""

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import convolve2d

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)


def read_rgb(path):
    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def to_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255)


def to_uint8(tensor):
    array = tensor.detach().float().squeeze(0).clamp(0, 1).cpu().numpy()
    return np.round(array.transpose(1, 2, 0) * 255).astype(np.uint8)


def write_rgb(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def flip_pad(tensor, window):
    height, width = tensor.shape[-2:]
    target_h = ((height + window - 1) // window) * window
    target_w = ((width + window - 1) // window) * window
    # Repeated concatenation also supports inputs smaller than one window.
    while tensor.shape[-2] < target_h:
        tensor = torch.cat((tensor, tensor.flip(-2)), dim=-2)
    tensor = tensor[..., :target_h, :]
    while tensor.shape[-1] < target_w:
        tensor = torch.cat((tensor, tensor.flip(-1)), dim=-1)
    return tensor[..., :target_w]


def y_channel(array):
    # Match single(rgb2ycbcr(im2double(rgb))) * 255 in the MATLAB evaluator.
    y = (16.0 + np.dot(array.astype(np.float64) / 255.0, [65.481, 128.553, 24.966])) / 255.0
    return (y.astype(np.float32) * np.float32(255.0)).astype(np.float64)


def psnr_ssim(prediction, target, scale):
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape}, {target.shape}")
    x = y_channel(prediction)[scale:-scale, scale:-scale]
    y = y_channel(target)[scale:-scale, scale:-scale]
    if min(x.shape) < 11:
        raise ValueError("The shaved image must be at least 11 x 11 pixels for SSIM.")
    mse = np.mean((x - y) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    gaussian = np.exp(-(np.arange(-5, 6, dtype=np.float64) ** 2) / (2 * 1.5**2))
    kernel = np.outer(gaussian, gaussian)
    kernel /= kernel.sum()
    mu_x = convolve2d(x, kernel, mode="valid")
    mu_y = convolve2d(y, kernel, mode="valid")
    var_x = convolve2d(x * x, kernel, mode="valid") - mu_x**2
    var_y = convolve2d(y * y, kernel, mode="valid") - mu_y**2
    cov = convolve2d(x * y, kernel, mode="valid") - mu_x * mu_y
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(psnr), float(ssim.mean())
