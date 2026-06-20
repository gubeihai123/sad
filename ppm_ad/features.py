from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


class WideResNetPatchExtractor(nn.Module):
    """Frozen ImageNet features aligned to a common patch grid."""

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")) -> None:
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.layers = tuple(layers)
        self.body = create_feature_extractor(backbone, return_nodes={x: x for x in self.layers})
        self.body.eval()
        for parameter in self.body.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.body(images)
        maps = [outputs[name] for name in self.layers]
        target_hw = maps[0].shape[-2:]
        maps = [F.interpolate(x, target_hw, mode="bilinear", align_corners=False) for x in maps]
        fused = torch.cat(maps, dim=1)
        fused = F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
        return fused.permute(0, 2, 3, 1).contiguous()


@dataclass
class RandomProjector:
    output_dim: int = 128
    seed: int = 0
    max_fit_patches: int = 50000

    def fit(self, features: np.ndarray) -> "RandomProjector":
        rng = np.random.default_rng(self.seed)
        if len(features) > self.max_fit_patches:
            features = features[rng.choice(len(features), self.max_fit_patches, replace=False)]
        self.model = GaussianRandomProjection(n_components=self.output_dim, random_state=self.seed)
        self.model.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        projected = self.model.transform(features).astype(np.float32, copy=False)
        norm = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norm, 1e-12)


@torch.inference_mode()
def extract_loader(loader, extractor: nn.Module, device: torch.device):
    feature_maps, masks, labels, paths, defects = [], [], [], [], []
    extractor.eval()
    for batch in loader:
        fmap = extractor(batch["image"].to(device)).cpu().numpy()
        feature_maps.append(fmap)
        masks.append(batch["mask"].numpy())
        labels.append(batch["label"].numpy())
        paths.extend(batch["path"])
        defects.extend(batch["defect"])
    return {
        "features": np.concatenate(feature_maps),
        "masks": np.concatenate(masks),
        "labels": np.concatenate(labels),
        "paths": paths,
        "defects": defects,
    }


