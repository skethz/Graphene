import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.build import build_model, load_weights, read_config
from models.graphene import IsingRefinementBlock
from tools.images import flip_pad, psnr_ssim, write_rgb
from tools.train import PairedDataset, batch_layout, optimizer_partition, make_optimizers, set_learning_rate

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def test_paper_recipe_and_global_batch(self):
        for path in (ROOT / "configs").glob("graphene*.json"):
            config = read_config(path)
            if "training" not in config:
                continue
            settings = config["training"]
            self.assertEqual((settings["lr_patch_size"], settings["global_batch_size"], settings["total_steps"]), (128, 32, 500000))
            self.assertEqual(settings["milestones"], [250000, 400000, 450000, 475000])
            self.assertEqual(batch_layout(32, 16, None), (2, 1))
            self.assertEqual(batch_layout(32, 1, 2), (2, 16))
        with self.assertRaises(ValueError):
            batch_layout(32, 3, None)

    def test_paired_crops_and_augmentation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            lr = np.random.default_rng(0).integers(0, 256, (150, 150, 3), dtype=np.uint8)
            hr = lr.repeat(2, axis=0).repeat(2, axis=1)
            write_rgb(root / "lr/a.png", lr)
            write_rgb(root / "hr/a.png", hr)
            data = PairedDataset([root / "hr"], [root / "lr"], 2, 128)
            x, y = data[0]
            self.assertEqual(tuple(x.shape), (3, 128, 128))
            self.assertEqual(tuple(y.shape), (3, 256, 256))
            torch.testing.assert_close(y, x.repeat_interleave(2, -1).repeat_interleave(2, -2), rtol=0, atol=0)

    def test_optimizer_partition_schedule_and_strict_loading(self):
        config = read_config(ROOT / "configs/graphene-lightweight_x2.json")
        model = build_model(config)
        muon, adam = optimizer_partition(model)
        self.assertTrue(any(name.endswith("ising.to_h.weight") for name, _ in muon))
        self.assertTrue(any(name == "conv_first.weight" for name, _ in adam))
        self.assertTrue(all(name.startswith("layers.") and p.ndim == 2 for name, p in muon))
        self.assertEqual(len(muon) + len(adam), len(list(model.parameters())))
        optimizers = make_optimizers(model, config["training"])
        self.assertEqual(set_learning_rate(optimizers, config["training"], 249999), 2e-4)
        self.assertEqual(set_learning_rate(optimizers, config["training"], 250000), 1e-4)
        self.assertEqual(set_learning_rate(optimizers, config["training"], 475000), 1.25e-5)
        with tempfile.TemporaryDirectory() as directory:
            state = {k: v for k, v in model.state_dict().items() if not k.endswith(".attn_mask")}
            path = Path(directory) / "weights.pth"
            torch.save(state, path)
            load_weights(model, path)
            del state["conv_first.weight"]
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_weights(model, path)

    def test_irb_equations(self):
        torch.manual_seed(1)
        block = IsingRefinementBlock(6, 4, dict(K=3, beta=1, gamma=1, gate_scale=1, signed=True, symmetric=True, topk=0)).double()
        x = torch.randn(2, 16, 6, dtype=torch.float64)
        logits = torch.randn(2, 2, 16, 4, dtype=torch.float64)
        mask = torch.zeros(2, 16, 16, dtype=torch.bool)
        mask[:, :4, 8:] = True
        coupling = logits.mean(1).index_select(-1, block.lift_index) / 4
        coupling = coupling.masked_fill(mask, 0)
        coupling = coupling - coupling.mean(-1, keepdim=True)
        coupling = (coupling + coupling.transpose(-1, -2)) / 2
        coupling.diagonal(dim1=-2, dim2=-1).zero_()
        state = torch.tanh(block.to_m0(x))
        for _ in range(3):
            state = torch.tanh(block.to_h(x) + coupling @ state)
        expected = x + torch.sigmoid(state) * block.refine(x)
        torch.testing.assert_close(block(x, logits, mask), expected)
        self.assertTrue(bool((coupling[mask] != 0).any()))

    def test_padding_and_metric_identity(self):
        x = torch.arange(15).reshape(1, 1, 3, 5)
        y = flip_pad(x, 16)
        self.assertEqual(tuple(y.shape), (1, 1, 16, 16))
        torch.testing.assert_close(y[..., :3, :5], x)
        image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        psnr, ssim = psnr_ssim(image, image, 4)
        self.assertEqual(psnr, float("inf"))
        self.assertAlmostEqual(ssim, 1)


if __name__ == "__main__":
    unittest.main()
