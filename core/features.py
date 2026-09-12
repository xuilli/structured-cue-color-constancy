from __future__ import annotations

from typing import List

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def extract_statistical_features(
    image: torch.Tensor,
    dark_threshold: float,
    saturation_threshold: float,
    use_topk_extreme_statistics: bool = False,
    topk_extreme_k: int = 20,
    use_robust_maximum_statistics: bool = False,
    robust_maximum_k: int = 20,
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Expected a three-channel CHW image, got {tuple(image.shape)}"
        )
    if topk_extreme_k <= 0:
        raise ValueError("topk_extreme_k must be positive")
    if robust_maximum_k <= 0:
        raise ValueError("robust_maximum_k must be positive")

    pixels = image.reshape(3, -1)
    mask = (pixels.mean(dim=0) > dark_threshold) & torch.all(
        pixels < saturation_threshold, dim=0
    )
    valid_pixels = pixels[:, mask] if mask.any() else pixels
    mean_value = valid_pixels.mean(dim=-1)

    if not mask.any():
        features = mean_value.repeat(4, 1)
        chromaticity = features / (features.sum(dim=-1, keepdim=True) + EPS)
        return chromaticity[:, [0, 2]].flatten()

    sums = valid_pixels.sum(dim=0)

    if use_robust_maximum_statistics:
        max_k = min(int(robust_maximum_k), int(valid_pixels.shape[1]))
        maximum = torch.topk(valid_pixels, k=max_k, dim=-1, largest=True).values.mean(
            dim=-1
        )
    else:
        maximum = valid_pixels.max(dim=-1).values

    if not use_topk_extreme_statistics:
        brightest = valid_pixels[:, torch.argmax(sums)]
        darkest = valid_pixels[:, torch.argmin(sums)]
        features = torch.stack([brightest, maximum, mean_value, darkest], dim=0)
        chromaticity = features / (features.sum(dim=-1, keepdim=True) + EPS)
        return chromaticity[:, [0, 2]].flatten()

    k = min(int(topk_extreme_k), int(valid_pixels.shape[1]))
    bright_idx = torch.topk(sums, k=k, largest=True).indices
    dark_idx = torch.topk(sums, k=k, largest=False).indices
    per_pixel_chrom = valid_pixels.transpose(0, 1)
    per_pixel_chrom = per_pixel_chrom / (
        per_pixel_chrom.sum(dim=-1, keepdim=True) + EPS
    )
    brightest_chrom = per_pixel_chrom[bright_idx].mean(dim=0)
    darkest_chrom = per_pixel_chrom[dark_idx].mean(dim=0)
    maximum_chrom = maximum / (maximum.sum() + EPS)
    mean_chrom = mean_value / (mean_value.sum() + EPS)
    chromaticity = torch.stack(
        [brightest_chrom, maximum_chrom, mean_chrom, darkest_chrom], dim=0
    )
    return chromaticity[:, [0, 2]].flatten()


class EPCCFeatureExtractor(nn.Module):
    def __init__(
        self,
        dark_threshold: float,
        saturation_threshold: float,
        edge_operator: str = "sobel",
        use_topk_extreme_statistics: bool = False,
        topk_extreme_k: int = 20,
        use_robust_maximum_statistics: bool = False,
        robust_maximum_k: int = 20,
    ) -> None:
        super().__init__()
        if edge_operator not in {"sobel", "scharr"}:
            raise ValueError("edge_operator must be 'sobel' or 'scharr'")
        self.dark_threshold = dark_threshold
        self.saturation_threshold = saturation_threshold
        self.edge_operator = edge_operator
        self.use_topk_extreme_statistics = bool(use_topk_extreme_statistics)
        if topk_extreme_k <= 0:
            raise ValueError("topk_extreme_k must be positive")
        self.topk_extreme_k = int(topk_extreme_k)
        self.use_robust_maximum_statistics = bool(use_robust_maximum_statistics)
        if robust_maximum_k <= 0:
            raise ValueError("robust_maximum_k must be positive")
        self.robust_maximum_k = int(robust_maximum_k)

        if edge_operator == "scharr":
            kernel_x = (
                torch.tensor(
                    [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
                    dtype=torch.float32,
                )
                / 32.0
            )
            kernel_y = (
                torch.tensor(
                    [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
                    dtype=torch.float32,
                )
                / 32.0
            )
            edge_kernel_x = kernel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            edge_kernel_y = kernel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            self.register_buffer("edge_kernel_x", edge_kernel_x, persistent=False)
            self.register_buffer("edge_kernel_y", edge_kernel_y, persistent=False)

    def _edge_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        if self.edge_operator == "sobel":
            return kornia.filters.sobel(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(padded, self.edge_kernel_x, groups=3)
        gy = F.conv2d(padded, self.edge_kernel_y, groups=3)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    @torch.no_grad()
    def _extract_one(self, image: torch.Tensor) -> torch.Tensor:
        base = extract_statistical_features(
            image,
            self.dark_threshold,
            self.saturation_threshold,
            use_topk_extreme_statistics=self.use_topk_extreme_statistics,
            topk_extreme_k=self.topk_extreme_k,
            use_robust_maximum_statistics=self.use_robust_maximum_statistics,
            robust_maximum_k=self.robust_maximum_k,
        )
        edge_image = self._edge_magnitude(image.unsqueeze(0)).squeeze(0)
        edge = extract_statistical_features(
            edge_image,
            self.dark_threshold,
            float("inf"),
            use_topk_extreme_statistics=self.use_topk_extreme_statistics,
            topk_extreme_k=self.topk_extreme_k,
            use_robust_maximum_statistics=self.use_robust_maximum_statistics,
            robust_maximum_k=self.robust_maximum_k,
        )
        return torch.cat([base, edge], dim=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected a three-channel BCHW image batch, got {tuple(images.shape)}"
            )
        outputs: List[torch.Tensor] = [self._extract_one(image) for image in images]
        return torch.stack(outputs, dim=0)
