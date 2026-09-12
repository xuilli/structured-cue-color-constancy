from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

EPS = 1e-8


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def detect_layout(root: Path) -> str:
    if (root / "demosaiced").is_dir() and (root / "gt").is_dir():
        return "png_json"
    if (root / "numpy_data").is_dir() and (root / "numpy_labels").is_dir():
        return "npy_npy"
    raise FileNotFoundError(
        f"Unsupported dataset layout under {root}. Expected demosaiced/ + gt/ "
        "or numpy_data/ + numpy_labels/."
    )


def normalize_illuminant(illuminant: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    illuminant = torch.as_tensor(illuminant, dtype=torch.float32).flatten()
    if illuminant.numel() != 3:
        raise ValueError(
            f"Expected three illuminant channels, got {tuple(illuminant.shape)}"
        )
    illuminant = torch.nan_to_num(illuminant, nan=0.0, posinf=0.0, neginf=0.0)
    illuminant = illuminant.clamp_min(0.0)
    total = illuminant.sum()
    if total <= eps:
        return torch.full((3,), 1.0 / 3.0, dtype=torch.float32)
    return illuminant / (total + eps)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(image, dtype=torch.float32)
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    maximum = float(image.max()) if image.numel() else 0.0
    if maximum <= 1.0:
        return image
    if maximum <= 255.0:
        image = image / 255.0
    elif maximum <= 65535.0:
        image = image / 65535.0
    else:
        image = image / (maximum + EPS)
    return image.clamp(0.0, 1.0)


def load_label(path: Path, layout: str) -> torch.Tensor:
    if layout == "png_json":
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)["white_point"]
    else:
        value = np.load(path).astype(np.float32)
    return normalize_illuminant(torch.as_tensor(value, dtype=torch.float32))


def load_image(path: Path, layout: str) -> torch.Tensor:
    if layout == "png_json":
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
    else:
        array = np.load(path).astype(np.float32)
        if array.ndim != 3:
            raise ValueError(
                f"Expected a three-dimensional image array, got {array.shape}"
            )
        image = torch.from_numpy(array)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        elif image.shape[0] != 3:
            raise ValueError(f"Cannot identify RGB channel dimension in {array.shape}")
    return normalize_image(image.contiguous())


def pair_samples(root: Path, layout: str) -> List[Sample]:
    if layout == "png_json":
        image_paths = list((root / "demosaiced").glob("*.png"))
        label_paths = list((root / "gt").glob("*.json"))
    else:
        image_paths = list((root / "numpy_data").glob("*.npy"))
        label_paths = list((root / "numpy_labels").glob("*.npy"))
    images = {path.stem: path for path in image_paths}
    labels = {path.stem: path for path in label_paths}
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())[:5]
        missing_images = sorted(labels.keys() - images.keys())[:5]
        raise RuntimeError(
            f"Image/label stem mismatch. Missing labels={missing_labels}, "
            f"missing images={missing_images}"
        )
    return [Sample(images[name], labels[name]) for name in sorted(images)]


def build_fold_indices(
    num_samples: int, num_folds: int, seed: int
) -> Tuple[List[List[int]], List[List[int]]]:
    if not 2 <= num_folds <= num_samples:
        raise ValueError("num_folds must be between 2 and num_samples")
    all_indices = np.arange(num_samples)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    training_folds: List[List[int]] = []
    validation_folds: List[List[int]] = []
    for train_indices, validation_indices in kfold.split(all_indices):
        training_folds.append(train_indices.tolist())
        validation_folds.append(validation_indices.tolist())
    return training_folds, validation_folds


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def random_crop_and_resize(
    image: torch.Tensor, size: int, scale: float
) -> torch.Tensor:
    _, height, width = image.shape
    crop_height = max(1, int(height * scale))
    crop_width = max(1, int(width * scale))
    top = random.randint(0, height - crop_height) if height > crop_height else 0
    left = random.randint(0, width - crop_width) if width > crop_width else 0
    crop = image[:, top : top + crop_height, left : left + crop_width]
    return resize_image(crop, size)


def sample_nearby_illuminant(center: torch.Tensor, radius: float) -> torch.Tensor:
    center = normalize_illuminant(center)
    center_r, center_g = float(center[0]), float(center[1])
    for _ in range(1000):
        red = random.uniform(max(0.0, center_r - radius), min(1.0, center_r + radius))
        green = random.uniform(
            max(0.0, center_g - radius), min(1.0 - red, center_g + radius)
        )
        if (red - center_r) ** 2 + (green - center_g) ** 2 <= radius**2:
            return normalize_illuminant(torch.tensor([red, green, 1.0 - red - green]))
    return center


def apply_awb_augmentation(
    image: torch.Tensor,
    source_illuminant: torch.Tensor,
    target_illuminant: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source = normalize_illuminant(source_illuminant)
    target = normalize_illuminant(target_illuminant)
    scale = target / (source + EPS)
    augmented = (normalize_image(image) * scale.view(3, 1, 1)).clamp(0.0, 1.0)
    return augmented, target


class PureColorDataset(Dataset):
    def __init__(self, cfg, mode: str, fold_index: int) -> None:
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")

        self.cfg = cfg
        self.mode = mode
        self.root = Path(cfg.data_root)
        self.layout = detect_layout(self.root)
        self.samples = pair_samples(self.root, self.layout)
        train_folds, val_folds = build_fold_indices(
            len(self.samples), cfg.fold_num, cfg.random_seed
        )
        self.indices = (
            train_folds[fold_index] if mode == "train" else val_folds[fold_index]
        )
        self.labels = [
            load_label(self.samples[sample_index].label_path, self.layout)
            for sample_index in self.indices
        ]

        self.illuminant_pool: List[torch.Tensor] = []
        if mode == "train" and cfg.use_awb_aug:
            self.illuminant_pool = [
                load_label(sample.label_path, self.layout) for sample in self.samples
            ]
            if not self.illuminant_pool:
                raise RuntimeError("Global illuminant pool is empty for AWB-Aug.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = self.samples[self.indices[index]]
        image = load_image(sample.image_path, self.layout)
        target = self.labels[index].clone()

        if self.mode == "train" and self.cfg.use_awb_aug:
            if random.random() < self.cfg.awb_aug_prob:
                center = random.choice(self.illuminant_pool)
                new_target = sample_nearby_illuminant(center, self.cfg.awb_aug_radius)
                image, target = apply_awb_augmentation(image, target, new_target)

        if self.mode == "train" and self.cfg.use_random_crop:
            image = random_crop_and_resize(
                image, self.cfg.image_size, self.cfg.train_crop_scale
            )
        else:
            image = resize_image(image, self.cfg.image_size)

        return {
            "image": image.contiguous(),
            "target": target.contiguous(),
            "name": sample.image_path.stem,
        }
