from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def angular_error_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_unit = F.normalize(prediction, dim=-1)
    target_unit = F.normalize(target, dim=-1)
    dot = torch.sum(prediction_unit * target_unit, dim=-1).clamp(-0.999999, 0.999999)
    return torch.rad2deg(torch.acos(dot))


def angular_error_mean(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return angular_error_batch(prediction, target).mean()


def official_summary(errors: Iterable[float]) -> Dict[str, float]:
    values = np.sort(np.asarray(list(errors), dtype=np.float64))
    if values.size < 4:
        raise ValueError("At least 4 samples are required to compute quartile metrics.")
    middle_1 = (len(values) + 1) // 2 - 1
    middle_2 = (len(values) + 2) // 2 - 1
    quarter_count = len(values) // 4
    median = (values[middle_1] + values[middle_2]) / 2
    first_quartile = values[quarter_count]
    third_quartile = values[-quarter_count]
    return {
        "mean": float(values.mean()),
        "median": float(median),
        "trimean": float((first_quartile + 2 * median + third_quartile) / 4),
        "best25": float(values[:quarter_count].mean()),
        "worst25": float(values[-quarter_count:].mean()),
    }
