from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class ExperimentConfig:
    data_root: Path = Path(r"D:/color constancy/datasets/pure_color_v2")
    output_root: Path = Path("outputs")
    experiment_name: str = "ours"

    fold_num: int = 3
    fold_index: int = 0
    run_all_folds: bool = True
    random_seed: int = 10

    image_size: int = 256
    use_random_crop: bool = True
    train_crop_scale: float = 0.7
    use_awb_aug: bool = True
    awb_aug_prob: float = 0.5
    awb_aug_radius: float = 0.02

    batch_size: int = 32
    validation_batch_size: int = 32
    num_workers: int = 4
    epochs: int = 1000
    learning_rate: float = 0.002
    optimizer: str = "adamw"
    weight_decay: float = 0.001
    scheduler_end_factor: float = 0.1
    drop_last_train_batch: bool = True

    dark_threshold: float = 0.02
    saturation_threshold: float = 0.98
    edge_operator: str = "scharr"

    use_topk_extreme_statistics: bool = True
    topk_extreme_k: int = 20

    use_robust_maximum_statistics: bool = True
    robust_maximum_k: int = 20

    interaction_type: str = "scr"
    fastkan_basis: str = "b_spline"

    token_dim: int = 24
    transformer_num_heads: int = 4

    kan_hidden_dims: Tuple[int, ...] = (8, 8, 8)

    rbf_num_centers: int = 8

    bspline_grid_intervals: int = 5
    bspline_order: int = 2

    fastkan_grid_min: float = -2.0
    fastkan_grid_max: float = 2.0
    fastkan_use_base_update: bool = True
    fastkan_use_layernorm: bool = True
    fastkan_base_activation: str = "leaky_relu"
    fastkan_spline_weight_init_scale: float = 0.1
    leaky_relu_slope: float = 0.01

    device: str | None = None

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            requested = torch.device(self.device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return requested
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _validated_interaction(self) -> str:
        interaction = self.interaction_type.lower().strip()
        if interaction not in {"none", "scr", "transformer"}:
            raise ValueError(
                "interaction_type must be 'none', 'scr', or 'transformer', "
                f"got {self.interaction_type!r}"
            )
        if interaction == "transformer":
            if self.transformer_num_heads <= 0:
                raise ValueError("transformer_num_heads must be positive")
            if self.token_dim % self.transformer_num_heads != 0:
                raise ValueError(
                    "token_dim must be divisible by transformer_num_heads "
                    "when interaction_type='transformer'"
                )
        return interaction

    def _validated_basis(self) -> str:
        basis = self.fastkan_basis.lower().strip()
        if basis not in {"b_spline", "rbf"}:
            raise ValueError(
                "fastkan_basis must be 'b_spline' or 'rbf', "
                f"got {self.fastkan_basis!r}"
            )
        return basis

    def validated_hidden_dims(self) -> Tuple[int, ...]:
        dims = tuple(int(value) for value in self.kan_hidden_dims)
        if len(dims) == 0:
            raise ValueError("kan_hidden_dims must contain at least one width")
        if any(value <= 0 for value in dims):
            raise ValueError("all kan_hidden_dims values must be positive")
        return dims

    def basis_count(self) -> int:
        basis = self._validated_basis()
        if basis == "b_spline":
            if self.bspline_grid_intervals < 1:
                raise ValueError("bspline_grid_intervals must be at least 1")
            if self.bspline_order < 0:
                raise ValueError("bspline_order cannot be negative")
            return self.bspline_grid_intervals + self.bspline_order

        if self.rbf_num_centers < 2:
            raise ValueError("rbf_num_centers must be at least 2")
        return self.rbf_num_centers

    def regression_input_dim(self) -> int:
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        return 8 * self.token_dim

    def experiment_dir(self) -> Path:
        name = self.experiment_name.strip()
        if not name:
            raise ValueError("experiment_name cannot be empty")
        return self.output_root / name


CFG = ExperimentConfig()
