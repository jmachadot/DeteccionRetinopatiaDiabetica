"""
Entrena un U-Net en DRIVE y lo evalúa en DRIVE-test, generando todos los
artefactos: métricas, ROC, ejemplos cualitativos y análisis por grosor de vaso.

Uso (ejemplos):
    python train.py --run-name baseline_drive
    python train.py --loss dice --run-name dice_drive
    python train.py --preprocess none --run-name baseline_no_clahe
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from src.data import get_drive_loaders, get_eval_loader
from src.models import build_model, count_params
from src.losses import build_loss
from src.engine import train_model, evaluate
from src.metrics import (
    plot_roc_pixel, save_qualitative_grid, format_metrics,
)
from src.failure_analysis import sensitivity_by_thickness, plot_sensitivity_by_thickness


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--loss", choices=["bce", "dice", "combined"], default=None)
    p.add_argument("--preprocess", choices=["none", "green", "clahe"], default=None)
    p.add_argument("--aug", dest="aug_strength",
                   choices=["none", "standard", "strong"], default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--tta", action="store_true", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def make_config(args) -> C.Config:
    cfg = C.Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    if args.run_name is None:
        cfg.run_name = f"d{cfg.depth}_{cfg.loss}_{cfg.preprocess}"
    return cfg


def run(cfg: C.Config, device, make_failure=True):
    run_dir = cfg.run_dir()
    cfg.save(run_dir / "config.json")
    print(f"\n=== RUN: {cfg.run_name} | device={device} ===")

    train_loader, val_loader, test_loader, info = get_drive_loaders(cfg)
    info["batch_size"] = cfg.batch_size
    print(f"[data] DRIVE: {info}")
    (run_dir / "data_info.json").write_text(json.dumps(info, indent=2))

    model = build_model(cfg)
    print(f"[model] params={count_params(model):,} | depth={cfg.depth} | base={cfg.base_channels}")
    criterion = build_loss(cfg)

    model, history, best_epoch = train_model(
        model, train_loader, val_loader, criterion, cfg, device, run_dir=run_dir
    )

    # Evaluación en DRIVE-test
    agg, per_image, dump = evaluate(model, test_loader, device,
                                    threshold=cfg.threshold, tta=cfg.tta)
    print(f"[TEST DRIVE] {format_metrics(agg)}")
    (run_dir / "test_metrics.json").write_text(json.dumps(agg, indent=2))

    # ROC global a nivel de píxel
    auc_global = plot_roc_pixel(dump["probs"], dump["gts"], dump["fovs"],
                                run_dir / "roc_pixel.png",
                                title=f"ROC píxel — {cfg.run_name}")
    print(f"[TEST DRIVE] AUC global (píxel) = {auc_global:.4f}")

    # Ejemplos cualitativos
    preds = [(p >= cfg.threshold).astype(np.uint8) for p in dump["probs"]]
    save_qualitative_grid(dump["imgs"], dump["gts"], preds, dump["fovs"],
                          dump["names"], run_dir / "qualitative.png", n=4)

    # Análisis de fallos por grosor
    if make_failure:
        by_thick = sensitivity_by_thickness(dump["probs"], dump["gts"], dump["fovs"],
                                            threshold=cfg.threshold)
        (run_dir / "sensitivity_by_thickness.json").write_text(json.dumps(by_thick, indent=2))
        plot_sensitivity_by_thickness(by_thick,
                                      run_dir / "sensitivity_by_thickness.png",
                                      title=f"Sensibilidad por grosor — {cfg.run_name}")
        print(f"[fallos] {by_thick}")

    return model, agg


if __name__ == "__main__":
    args = parse_args()
    cfg = make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, device)
