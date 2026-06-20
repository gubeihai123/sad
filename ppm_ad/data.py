from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
)


def image_transform(image_size: int) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def mask_transform(image_size: int) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])


def apply_drift(image: Image.Image, level: float) -> Image.Image:
    """Deterministic gradual production-like drift: brightness, contrast and blur."""
    if level <= 0:
        return image
    image = ImageEnhance.Brightness(image).enhance(1.0 + 0.35 * level)
    image = ImageEnhance.Contrast(image).enhance(1.0 - 0.20 * level)
    return image.filter(ImageFilter.GaussianBlur(radius=1.5 * level))


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_size: int = 256,
        shots: Optional[int] = None,
        seed: int = 0,
        drift_level: float = 0.0,
    ) -> None:
        if category not in MVTEC_CATEGORIES:
            raise ValueError(f"Unknown MVTec category: {category}")
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.drift_level = drift_level
        self.xform = image_transform(image_size)
        self.mask_xform = mask_transform(image_size)

        split_dir = self.root / category / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"MVTec category not found: {split_dir}. See README dataset setup."
            )
        samples: list[tuple[Path, str]] = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(defect_dir.glob("*.png")):
                samples.append((path, defect_dir.name))
        if split == "train":
            samples = [x for x in samples if x[1] == "good"]
            rng = random.Random(seed)
            rng.shuffle(samples)
            if shots is not None:
                if len(samples) < shots:
                    raise ValueError(f"{category} has only {len(samples)} training images, requested {shots}")
                samples = samples[:shots]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, defect = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = apply_drift(image, self.drift_level)
        is_anomaly = int(defect != "good")
        if is_anomaly:
            mask_path = (
                self.root / self.category / "ground_truth" / defect /
                f"{image_path.stem}_mask.png"
            )
            mask = Image.open(mask_path).convert("L")
            mask_tensor = (self.mask_xform(mask).float() / 255.0 > 0.5).float()
        else:
            size = self.xform.transforms[0].size
            mask_tensor = torch.zeros((1, size[0], size[1]), dtype=torch.float32)
        return {
            "image": self.xform(image),
            "mask": mask_tensor,
            "label": torch.tensor(is_anomaly, dtype=torch.long),
            "path": str(image_path),
            "defect": defect,
        }


