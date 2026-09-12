from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PureColorDataset
from .metrics import angular_error_batch, angular_error_mean, official_summary
from .model import EPCC


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def build_loaders(cfg, fold_index: int) -> Tuple[DataLoader, DataLoader]:
    train_set = PureColorDataset(cfg, "train", fold_index)
    val_set = PureColorDataset(cfg, "val", fold_index)
    pin_memory = cfg.resolved_device().type == "cuda"
    generator = torch.Generator().manual_seed(cfg.random_seed)
    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.num_workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.drop_last_train_batch,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.validation_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def build_optimizer_and_scheduler(cfg, model: torch.nn.Module):
    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=cfg.scheduler_end_factor,
        total_iters=cfg.epochs,
    )
    return optimizer, scheduler


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    errors: List[float] = []
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            predictions = model(images)
            batch_errors = angular_error_batch(predictions, targets)
            for name, prediction, target, error in zip(
                batch["name"], predictions.cpu(), targets.cpu(), batch_errors.cpu()
            ):
                error_value = float(error.item())
                errors.append(error_value)
                records.append(
                    {
                        "name": name,
                        "prediction": [float(value) for value in prediction.tolist()],
                        "target": [float(value) for value in target.tolist()],
                        "angular_error": error_value,
                    }
                )
    return official_summary(errors), records


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def train_one_fold(cfg, fold_index: int) -> Dict[str, object]:
    seed_everything(cfg.random_seed)
    device = cfg.resolved_device()
    model = EPCC(cfg).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    train_loader, val_loader = build_loaders(cfg, fold_index)
    fold_dir = cfg.experiment_dir() / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_json(fold_dir / "config.json", asdict(cfg))

    best_mean = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = angular_error_mean(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.shape[0]
            total_count += images.shape[0]
        scheduler.step()
        validation_metrics, _ = validate(model, val_loader, device)
        train_loss = total_loss / max(1, total_count)
        history.append(
            {
                "epoch": epoch,
                "train_mean_angular_error": train_loss,
                "val_mean_angular_error": validation_metrics["mean"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[fold {fold_index}] {epoch:03d}/{cfg.epochs} "
            f"train={train_loss:.4f}° val={validation_metrics['mean']:.4f}°"
        )
        if validation_metrics["mean"] < best_mean:
            best_mean = validation_metrics["mean"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": validation_metrics,
                },
                fold_dir / "best.pth",
            )

    if best_epoch < 0:
        raise RuntimeError(
            "No best checkpoint was saved. Check the epoch count and validation procedure."
        )
    save_json(fold_dir / "history.json", history)
    return {"fold": fold_index, "best_epoch": best_epoch, "best_mean": best_mean}


def evaluate_one_fold(cfg, fold_index: int) -> Dict[str, object]:
    device = cfg.resolved_device()
    model = EPCC(cfg)
    checkpoint_path = cfg.experiment_dir() / f"fold_{fold_index}" / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    _, val_loader = build_loaders(cfg, fold_index)
    metrics, records = validate(model, val_loader, device)
    result = {"fold": fold_index, "metrics": metrics, "samples": records}
    save_json(cfg.experiment_dir() / f"fold_{fold_index}" / "test_results.json", result)
    return result


def summarize_folds(cfg, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ["mean", "median", "trimean", "best25", "worst25"]
    compact_folds = [
        {"fold": result["fold"], "metrics": result["metrics"]}
        for result in fold_results
    ]
    average = {
        name: float(np.mean([result["metrics"][name] for result in compact_folds]))
        for name in metric_names
    }
    summary = {"folds": compact_folds, "average": average}
    save_json(cfg.experiment_dir() / "cross_validation_summary.json", summary)
    return summary
