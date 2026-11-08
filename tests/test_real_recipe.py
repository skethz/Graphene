from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from models.build import read_config
from tools.degradations.bsrgan import degradation_bsrgan, shift_pixel
from tools.images import write_rgb
from tools.train_real import BlindSRDataset, optimizer_step

ROOT = Path(__file__).resolve().parents[1]


class RealRecipeTests(unittest.TestCase):
    def test_swinir_stage_settings(self):
        psnr = read_config(ROOT / "configs/train_graphene-realworld_x4_psnr.json")["training"]
        gan = read_config(ROOT / "configs/train_graphene-realworld_x4_gan.json")["training"]
        for settings in (psnr, gan):
            self.assertEqual((settings["degradation"], settings["lr_patch_size"], settings["hr_patch_size"], settings["global_batch_size"]), ("bsrgan", 64, 256, 32))
            self.assertEqual(settings["adam_betas"], [0.9, 0.999])
            self.assertEqual(settings["precision"], "fp32")
        self.assertEqual(psnr["learning_rate"], 2e-4)
        self.assertEqual(psnr["milestones"], [500000, 800000, 900000, 950000, 1000000])
        self.assertEqual(gan["learning_rate"], 1e-4)
        self.assertEqual(gan["milestones"], [400000, 500000, 550000, 575000, 600000])
        self.assertEqual(gan["feature_layers"], [2, 7, 16, 25, 34])
        self.assertEqual(gan["feature_weights"], [0.1, 0.1, 1.0, 1.0, 1.0])
        self.assertEqual(gan["adversarial_weight"], 0.1)

    def test_bilinear_shift(self):
        y, x = np.mgrid[:7, :9].astype(np.float64)
        image = 2 * y + 3 * x
        shifted = shift_pixel(image, sf=4)
        expected = 2 * np.clip(y + 1.5, 0, 6) + 3 * np.clip(x + 1.5, 0, 8)
        np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-12)

    def test_degradation_and_dataset(self):
        image = np.random.default_rng(3).integers(0, 256, (260, 260, 3), dtype=np.uint8)
        settings = read_config(ROOT / "configs/train_graphene-realworld_x4_psnr.json")["training"]
        for seed in range(8):
            random.seed(seed)
            np.random.seed(seed)
            lr, hr = degradation_bsrgan(image[:256, :256].astype(np.float32) / 255, sf=4, lq_patchsize=64)
            self.assertEqual(lr.shape, (64, 64, 3))
            self.assertEqual(hr.shape, (256, 256, 3))
            self.assertTrue(np.isfinite(lr).all() and (lr >= 0).all() and (lr <= 1).all())
            np.testing.assert_array_equal(hr, image[:256, :256].astype(np.float32) / 255)
        with tempfile.TemporaryDirectory() as directory:
            write_rgb(Path(directory) / "image.png", image)
            lr, hr = BlindSRDataset([directory], settings)[0]
            self.assertEqual(tuple(lr.shape), (3, 64, 64))
            self.assertEqual(tuple(hr.shape), (3, 256, 256))
            self.assertEqual(lr.dtype, torch.float32)

    def test_gan_updates_both_models(self):
        torch.manual_seed(1)
        generator = nn.Conv2d(3, 3, 1)
        discriminator = nn.Conv2d(3, 1, 1)
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-4) for module in (generator, discriminator)]
        before = [next(module.parameters()).detach().clone() for module in (generator, discriminator)]
        settings = read_config(ROOT / "configs/train_graphene-realworld_x4_gan.json")["training"]
        batches = [(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)) for _ in range(2)]
        losses = optimizer_step(generator, discriminator, nn.L1Loss(), batches, optimizers, settings)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        for old, module in zip(before, (generator, discriminator)):
            self.assertFalse(torch.equal(old, next(module.parameters())))


if __name__ == "__main__":
    unittest.main()
