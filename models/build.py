"""Construct Graphene and load all learned weights strictly."""

import json
from pathlib import Path

import torch

from .graphene import Graphene


def read_config(path):
    return json.loads(Path(path).read_text())


def build_model(config, checkpoint=None, training=False, activation_checkpoint=False):
    options = dict(config["model"])
    if training:
        options["img_size"] = config["training"]["lr_patch_size"]
    options["use_checkpoint"] = activation_checkpoint
    model = Graphene(**options)
    model.fp32_io = training and options["upsampler"] in ("pixelshuffle", "pixelshuffledirect")
    if checkpoint is not None:
        load_weights(model, checkpoint)
    return model


def load_weights(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    state = dict(state)
    expected = model.state_dict()
    # Attention masks depend on the configured input size, not learned weights.
    # Reference checkpoints omit them; reconstruct them for this model instance.
    for key in list(state):
        if key.endswith(".attn_mask") and key not in expected:
            del state[key]
    for key, value in expected.items():
        if key.endswith(".attn_mask"):
            state[key] = value
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def prepare_masks(model, height, width):
    """Cache deterministic masks for fixed-shape efficiency measurements."""
    shared = {}
    for stage in model.layers:
        for block in stage.residual_group.blocks:
            block.input_resolution = (height, width)
            if block.shift_size:
                # Blocks with the same window and shift share one mask copy.
                key = (block.window_size, block.shift_size)
                if key not in shared:
                    shared[key] = [block.calculate_mask((height, width)).to(next(block.parameters()).device), None]
                block.attn_mask = shared[key][0]
                if block.ising is not None:
                    if shared[key][1] is None:
                        shared[key][1] = block.ising._lift_pair_mask(shared[key][0]).lt(0)
                    block.ising_pair_mask = shared[key][1]
