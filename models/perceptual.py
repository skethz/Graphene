"""VGG19 feature loss following SwinIR's KAIR real-SR training recipe."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(2, 7, 16, 25, 34), weights=(0.1, 0.1, 1, 1, 1)):
        super().__init__()
        if len(layers) != len(weights) or sorted(set(layers)) != list(layers):
            raise ValueError("VGG layer indices must be unique, increasing, and paired with weights.")
        self.layers = tuple(layers)
        self.weights = tuple(weights)
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:layers[-1] + 1]
        self.features.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract(self, image):
        features = []
        image = (image - self.mean) / self.std
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in self.layers:
                # Later VGG ReLUs are in-place; preserve each selected pre-ReLU map.
                features.append(image.clone())
        return features

    def forward(self, prediction, target):
        prediction_features = self.extract(prediction)
        with torch.no_grad():
            target_features = self.extract(target)
        return sum(weight * F.l1_loss(predicted, expected)
                   for weight, predicted, expected in zip(self.weights, prediction_features, target_features))
