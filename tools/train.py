"""DF2K training for the classical and lightweight Graphene models."""

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.build import build_model, read_config
from tools.images import EXTENSIONS, image_files, read_rgb, to_tensor


def find_target(relative, root, scale):
    relative = Path(relative)
    stems = [relative.stem]
    if "_LRBI_" in relative.stem:
        stems.append(relative.stem.replace("_LRBI_", "_HR_"))
    stripped = re.sub(rf"(?:_LRBI|_LR)?[xX]{scale}$", "", relative.stem)
    if stripped != relative.stem:
        stems.append(stripped)
    candidates = [root / relative.parent / (stem + suffix)
                  for stem in stems for suffix in sorted(EXTENSIONS)]
    matches = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(matches) != 1:
        raise ValueError(f"Expected one HR image for {relative} in {root}; found {matches}")
    return matches[0]


class ImageStore:
    """Read an image directory or an encoded-image LMDB with meta_info.txt."""

    def __init__(self, path):
        self.path = Path(path)
        self.is_lmdb = self.path.suffix == ".lmdb"
        self.env = None
        if self.is_lmdb:
            self.names = [line.split()[0] for line in (self.path / "meta_info.txt").read_text().splitlines() if line.strip()]
        else:
            self.names = [str(p.relative_to(self.path)) for p in image_files(self.path)]
        if not self.names:
            raise ValueError(f"No images found in {path}")

    def read(self, name):
        if not self.is_lmdb:
            return read_rgb(self.path / name)
        import lmdb
        if self.env is None:
            self.env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False)
        with self.env.begin(write=False) as transaction:
            value = transaction.get(str(Path(name).with_suffix("")).encode())
        if value is None:
            raise KeyError(f"Missing LMDB image {name}")
        image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Invalid encoded LMDB image {name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class PairedDataset(Dataset):
    def __init__(self, hr_roots, lr_roots, scale, lr_patch_size):
        if len(hr_roots) != len(lr_roots):
            raise ValueError("Supply the same number of HR and LR roots.")
        self.scale, self.patch = scale, lr_patch_size
        self.stores, self.pairs = [], []
        for hr_root, lr_root in zip(hr_roots, lr_roots):
            hr, lr = ImageStore(hr_root), ImageStore(lr_root)
            index = len(self.stores)
            self.stores.append((hr, lr))
            hr_names = set(hr.names)
            for name in lr.names:
                if name in hr_names:
                    target = name
                elif hr.is_lmdb:
                    stem = re.sub(rf"(?:_LRBI|_LR)?[xX]{scale}$", "", Path(name).stem)
                    matches = [candidate for candidate in hr.names if Path(candidate).stem == stem]
                    if len(matches) != 1:
                        raise ValueError(f"Ambiguous HR pairing for {name}")
                    target = matches[0]
                else:
                    target = str(find_target(name, hr.path, scale).relative_to(hr.path))
                self.pairs.append((index, target, name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        store_index, hr_name, lr_name = self.pairs[index]
        hr_store, lr_store = self.stores[store_index]
        hr, lr = hr_store.read(hr_name), lr_store.read(lr_name)
        if hr.shape[:2] != (lr.shape[0] * self.scale, lr.shape[1] * self.scale):
            raise ValueError(f"HR/LR alignment mismatch for {lr_name}: {hr.shape}, {lr.shape}")
        if min(lr.shape[:2]) < self.patch:
            raise ValueError(f"{lr_name} is smaller than the {self.patch}-pixel LR training crop.")
        y = random.randrange(lr.shape[0] - self.patch + 1)
        x = random.randrange(lr.shape[1] - self.patch + 1)
        lr = lr[y:y + self.patch, x:x + self.patch]
        y, x, size = y * self.scale, x * self.scale, self.patch * self.scale
        hr = hr[y:y + size, x:x + size]
        if random.random() < 0.5:
            lr, hr = np.flip(lr, axis=1), np.flip(hr, axis=1)
        rotation = random.randrange(4)
        lr, hr = np.rot90(lr, rotation), np.rot90(hr, rotation)
        return to_tensor(lr), to_tensor(hr)


def optimizer_partition(model):
    # Linear matrices in the shared LR encoder use Muon, including IRB maps.
    # Convolutions, biases, normalization, position tables, and history parameters use Adam.
    muon_ids = {id(module.weight) for name, module in model.named_modules()
                if name.startswith("layers.") and isinstance(module, torch.nn.Linear)}
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (muon if id(parameter) in muon_ids else adam).append((name, parameter))
    if not muon or not adam:
        raise ValueError("The Graphene optimizer requires both encoder matrices and other parameters.")
    return muon, adam


def make_optimizers(model, settings):
    muon, adam = optimizer_partition(model)
    common = dict(lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    optimizers = [
        torch.optim.Muon([p for _, p in muon], **common,
                         momentum=settings["muon_momentum"], nesterov=settings["muon_nesterov"],
                         ns_steps=settings["muon_ns_steps"], adjust_lr_fn=None),
        torch.optim.Adam([p for _, p in adam], **common, betas=tuple(settings["adam_betas"])),
    ]
    return optimizers


def set_learning_rate(optimizers, settings, completed_steps):
    multiplier = settings["gamma"] ** sum(completed_steps >= value for value in settings["milestones"])
    rate = settings["learning_rate"] * multiplier
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = rate
    return rate


@torch.no_grad()
def update_ema(ema, model, decay):
    for destination, source in zip(ema.parameters(), model.parameters()):
        destination.lerp_(source.detach(), 1 - decay)
    for destination, source in zip(ema.buffers(), model.buffers()):
        destination.copy_(source)


def batch_layout(global_batch, world_size, micro_batch):
    if global_batch % world_size:
        raise ValueError("Global batch size must be divisible by the number of GPUs.")
    per_rank = global_batch // world_size
    micro_batch = micro_batch or per_rank
    if micro_batch < 1 or per_rank % micro_batch:
        raise ValueError("Microbatch size must divide global_batch / world_size exactly.")
    return micro_batch, per_rank // micro_batch


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hr", nargs="+", required=True, help="DF2K HR image directories or LMDBs")
    parser.add_argument("--lr", nargs="+", required=True, help="Matching bicubic LR directories or LMDBs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Training state saved by this trainer")
    parser.add_argument("--micro-batch", type=int, help="Per-GPU batch; accumulation preserves the configured global batch")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Optional early stop for a short functional run")
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["training"]
    if settings["loss"] != "l1" or settings["precision"] != "bf16":
        raise ValueError("This trainer implements the paper's L1/BF16 recipe.")
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
    dataset = PairedDataset(args.hr, args.lr, config["model"]["upscale"], settings["lr_patch_size"])
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=settings["seed"], drop_last=True)
    loader = DataLoader(dataset, batch_size=micro_batch, sampler=sampler, num_workers=args.workers,
                        drop_last=True, pin_memory=True, worker_init_fn=seed_worker,
                        persistent_workers=args.workers > 0)
    if not len(loader):
        raise ValueError("The dataset is too small for the selected batch and GPU count.")
    model = build_model(config, training=True, activation_checkpoint=args.activation_checkpoint).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizers = make_optimizers(model, settings)
    step, epoch = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume configuration differs from the saved training recipe.")
        model.load_state_dict(state["params"], strict=True)
        ema.load_state_dict(state["params_ema"], strict=True)
        for optimizer, saved in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        step, epoch = state["step"], state["epoch"]
    train_model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(dict(global_batch=settings["global_batch_size"], world_size=world,
                              micro_batch=micro_batch, accumulation=accumulation,
                              lr_patch=settings["lr_patch_size"], steps=settings["total_steps"])), flush=True)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    stop = min(settings["total_steps"], args.max_steps or settings["total_steps"])
    while step < stop:
        rate = set_learning_rate(optimizers, settings, step)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for micro in range(accumulation):
            try:
                lr, hr = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                lr, hr = next(iterator)
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            sync = train_model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = train_model(lr)
                    loss = torch.nn.functional.l1_loss(prediction.float(), hr)
                (loss / accumulation).backward()
            total_loss += loss.detach() / accumulation
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        for optimizer in optimizers:
            optimizer.step()
        update_ema(ema, model, settings["ema_decay"])
        step += 1
        if rank == 0 and (step == 1 or step % 100 == 0 or step == stop):
            print(json.dumps(dict(step=step, l1=float(total_loss), learning_rate=rate)), flush=True)
        if step % settings["save_every"] == 0 or step == stop:
            if rank == 0:
                state = dict(params=model.state_dict(), params_ema=ema.state_dict(),
                             optimizers=[o.state_dict() for o in optimizers], step=step, epoch=epoch, config=config)
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
