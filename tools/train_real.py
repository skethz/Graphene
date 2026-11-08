"""SwinIR-style BSRGAN pretraining and GAN fine-tuning for graphene-realworld."""

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from models.build import build_model, load_weights, read_config
from models.discriminator import UNetDiscriminator
from models.perceptual import VGGPerceptualLoss
from tools.degradations.bsrgan import degradation_bsrgan
from tools.degradations.image_ops import augment_img
from tools.train import ImageStore, batch_layout, seed_worker, set_learning_rate, update_ema


class BlindSRDataset(Dataset):
    def __init__(self, roots, settings):
        if settings["degradation"] != "bsrgan":
            raise ValueError("The SwinIR real-SR recipe uses the original BSRGAN degradation.")
        self.stores = [ImageStore(root) for root in roots]
        self.images = [(index, name) for index, store in enumerate(self.stores) for name in store.names]
        self.hr_patch = settings["hr_patch_size"]
        self.lr_patch = settings["lr_patch_size"]
        if self.hr_patch != 4 * self.lr_patch:
            raise ValueError("Real-SR crops must have a scale factor of four.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        store, name = self.images[index]
        image = self.stores[store].read(name)
        height, width = image.shape[:2]
        if min(height, width) < self.hr_patch:
            raise ValueError(f"Training image {name} must be at least {self.hr_patch} pixels on each side.")
        top = random.randrange(height - self.hr_patch + 1)
        left = random.randrange(width - self.hr_patch + 1)
        image = image[top:top + self.hr_patch, left:left + self.hr_patch]
        mode = random.choice((0, 4)) if "face" in Path(name).stem else random.randrange(8)
        image = augment_img(image, mode=mode).astype(np.float32) / 255.0
        lr, hr = degradation_bsrgan(image, sf=4, lq_patchsize=self.lr_patch, isp_model=None)
        return tuple(torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1) for array in (lr, hr))


def initialize_discriminator(model, gain):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = getattr(module, "weight_orig", module.weight)
            torch.nn.init.orthogonal_(weight, gain=gain)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def adversarial_loss(prediction, real):
    target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
    return F.binary_cross_entropy_with_logits(prediction, target)


def optimizer_step(generator, discriminator, perceptual, batches, optimizers, settings):
    """Accumulate a global batch, then update G and D once each."""
    bare_discriminator = getattr(discriminator, "module", discriminator)
    if bare_discriminator is not None:
        bare_discriminator.requires_grad_(False)
    optimizer_g = optimizers[0]
    optimizer_g.zero_grad(set_to_none=True)
    count = len(batches)
    device = batches[0][0].device
    totals = {key: torch.zeros((), device=device) for key in ("pixel", "perceptual", "adversarial", "discriminator")}
    discriminator_inputs = []
    for index, (lr, hr) in enumerate(batches):
        context = generator.no_sync() if isinstance(generator, DistributedDataParallel) and index + 1 < count else nullcontext()
        with context:
            prediction = generator(lr)
            pixel = settings["pixel_weight"] * F.l1_loss(prediction, hr)
            loss = pixel
            totals["pixel"] += pixel.detach() / count
            if bare_discriminator is not None:
                feature = settings["perceptual_weight"] * perceptual(prediction, hr)
                adversarial = settings["adversarial_weight"] * adversarial_loss(bare_discriminator(prediction), True)
                loss = loss + feature + adversarial
                totals["perceptual"] += feature.detach() / count
                totals["adversarial"] += adversarial.detach() / count
                discriminator_inputs.append((prediction.detach(), hr))
            (loss / count).backward()
    if not all(bool(torch.isfinite(value)) for value in totals.values()):
        raise FloatingPointError("Nonfinite generator loss.")
    optimizer_g.step()
    if bare_discriminator is not None:
        bare_discriminator.requires_grad_(True)
        optimizers[1].zero_grad(set_to_none=True)
        for index, (prediction, hr) in enumerate(discriminator_inputs):
            # Separate real/fake backwards match KAIR and preserve spectral-norm buffers.
            context = discriminator.no_sync() if isinstance(discriminator, DistributedDataParallel) else nullcontext()
            with context:
                real_loss = adversarial_loss(discriminator(hr), True)
                (real_loss / count).backward()
            context = discriminator.no_sync() if isinstance(discriminator, DistributedDataParallel) and index + 1 < count else nullcontext()
            with context:
                fake_loss = adversarial_loss(discriminator(prediction.clone()), False)
                (fake_loss / count).backward()
            totals["discriminator"] += (real_loss.detach() + fake_loss.detach()) / count
        if not bool(torch.isfinite(totals["discriminator"])):
            raise FloatingPointError("Nonfinite discriminator loss.")
        optimizers[1].step()
    return {key: float(value) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hr", nargs="+", required=True, help="DIV2K, Flickr2K, and OST HR directories or LMDBs")
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--initialize", help="PSNR-stage generator/EMA weights for GAN fine-tuning")
    group.add_argument("--resume", help="Full training state from this stage")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Stop early for a functional run")
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["training"]
    if settings["stage"] not in ("psnr", "gan") or settings["optimizer"] != "adam" or settings["precision"] != "fp32":
        raise ValueError("Expected the SwinIR PSNR/GAN recipe with Adam and FP32 training.")
    gan = settings["stage"] == "gan"
    if gan and not (args.initialize or args.resume):
        parser.error("GAN fine-tuning requires --initialize with PSNR-stage weights, or --resume.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"] + rank)
    np.random.seed(settings["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    micro_batch, accumulation = batch_layout(settings["global_batch_size"], world, args.micro_batch)
    dataset = BlindSRDataset(args.hr, settings)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=settings["seed"], drop_last=True)
    loader = DataLoader(dataset, batch_size=micro_batch, sampler=sampler, num_workers=args.workers,
                        drop_last=True, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    if not len(loader):
        raise ValueError("The dataset is too small for the selected microbatch and GPU count.")
    model = build_model(config, training=True, activation_checkpoint=args.activation_checkpoint).to(device).train()
    if args.initialize:
        load_weights(model, args.initialize)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    discriminator, perceptual = None, None
    optimizer_g = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                   betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    optimizers = [optimizer_g]
    if gan:
        discriminator = UNetDiscriminator(input_nc=3, ndf=settings["discriminator_features"]).to(device).train()
        initialize_discriminator(discriminator, settings["discriminator_init_gain"])
        perceptual = VGGPerceptualLoss(settings["feature_layers"], settings["feature_weights"]).to(device).eval()
        optimizers.append(torch.optim.Adam(discriminator.parameters(), lr=settings["learning_rate"],
                                           betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"]))
    step, epoch = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume configuration differs from the saved stage recipe.")
        model.load_state_dict(state["params"], strict=True)
        ema.load_state_dict(state["params_ema"], strict=True)
        if gan:
            discriminator.load_state_dict(state["discriminator"], strict=True)
        for optimizer, saved in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        step, epoch = state["step"], state["epoch"]
    train_model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    train_discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank], broadcast_buffers=False) if gan and world > 1 else discriminator
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(dict(stage=settings["stage"], precision="fp32", global_batch=settings["global_batch_size"],
                              micro_batch=micro_batch, accumulation=accumulation, world_size=world)), flush=True)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    stop = min(settings["total_steps"], args.max_steps or settings["total_steps"])
    while step < stop:
        rate = set_learning_rate(optimizers, settings, step + 1)
        batches = []
        for _ in range(accumulation):
            try:
                lr, hr = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                lr, hr = next(iterator)
            batches.append((lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)))
        losses = optimizer_step(train_model, train_discriminator, perceptual, batches, optimizers, settings)
        update_ema(ema, model, settings["ema_decay"])
        step += 1
        if rank == 0 and (step == 1 or step % 200 == 0 or step == stop):
            print(json.dumps(dict(step=step, learning_rate=rate, **losses)), flush=True)
        if step % settings["save_every"] == 0 or step == stop:
            if rank == 0:
                state = dict(params=model.state_dict(), params_ema=ema.state_dict(),
                             optimizers=[optimizer.state_dict() for optimizer in optimizers],
                             step=step, epoch=epoch, config=config)
                if gan:
                    state["discriminator"] = discriminator.state_dict()
                temporary = output / "training.tmp"
                torch.save(state, temporary)
                temporary.replace(output / "training.pth")
                torch.save(ema.state_dict(), output / f"ema_{step}.pth")
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
