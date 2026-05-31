"""
Experimentos que cubren los entregables del enunciado:

  (A) Ablación arquitectónica / de entrenamiento (entregable 2):
      - Función de pérdida: BCE vs Dice vs combinada
      - Profundidad de la U-Net: 3 vs 4 etapas
  (B) Generalización entre datasets (entregable 4):
      Entrena en DRIVE, evalúa en DRIVE-test y CHASE_DB1; reporta la brecha.
  (C) Adaptación de dominio mediante CLAHE (entregable 6):
      Compara baseline (sin CLAHE) vs CLAHE; mide reducción de la brecha.

Cada experimento reutiliza `train.run`.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from src.data import get_eval_loader
from src.engine import evaluate
from src.metrics import format_metrics, plot_roc_pixel
from train import run as train_run


def _save_table(rows, out_path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"[tabla] {out_path}")


def _row(name, m, extra=None):
    r = {"experiment": name}
    if extra:
        r.update(extra)
    for k in ["sensitivity", "specificity", "f1", "auc_roc", "accuracy"]:
        r[k] = round(float(m.get(k, float("nan"))), 4)
    return r


def _override(base: C.Config, **kwargs) -> C.Config:
    cfg = C.Config(**{k: getattr(base, k) for k in base.__dataclass_fields__})
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# (A) Ablación
# --------------------------------------------------------------------------- #
def ablation_study(device, base: C.Config, out_dir: Path):
    rows = []
    configs = [
        ("loss=bce",      _override(base, loss="bce",      run_name="A_loss_bce")),
        ("loss=dice",     _override(base, loss="dice",     run_name="A_loss_dice")),
        ("loss=combined", _override(base, loss="combined", run_name="A_loss_combined")),
        ("depth=3",       _override(base, depth=3,         run_name="A_depth3")),
        ("depth=4",       _override(base, depth=4,         run_name="A_depth4")),
    ]
    for name, cfg in configs:
        _, agg = train_run(cfg, device, make_failure=False)
        rows.append(_row(name, agg))
    _save_table(rows, out_dir / "ablation.csv")
    return rows


# --------------------------------------------------------------------------- #
# (B) Generalización DRIVE -> CHASE_DB1
# --------------------------------------------------------------------------- #
def cross_dataset_eval(model, cfg, device, run_dir: Path,
                       target: str = "CHASEDB1") -> dict:
    """Evalúa un modelo entrenado en DRIVE sobre el dataset objetivo."""
    target_cfg = _override(cfg)  # mismo preprocesamiento y tamaño
    loader, info = get_eval_loader(target_cfg, target)
    agg, _, dump = evaluate(model, loader, device, threshold=cfg.threshold, tta=cfg.tta)
    out = run_dir / f"eval_{target}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({**agg, **info}, indent=2))
    plot_roc_pixel(dump["probs"], dump["gts"], dump["fovs"],
                   out / "roc_pixel.png",
                   title=f"ROC píxel — {target} (modelo entrenado en DRIVE)")
    from src.metrics import save_qualitative_grid
    preds = [(p >= cfg.threshold).astype(np.uint8) for p in dump["probs"]]
    save_qualitative_grid(dump["imgs"], dump["gts"], preds, dump["fovs"],
                          dump["names"], out / "qualitative.png", n=4)
    print(f"[{target}] {format_metrics(agg)}")
    return agg


def cross_dataset_study(device, base: C.Config, out_dir: Path):
    """Entrena una vez en DRIVE y evalúa en DRIVE-test y CHASE_DB1; reporta brecha."""
    cfg = _override(base, run_name="B_cross_baseline")
    model, agg_drive = train_run(cfg, device, make_failure=False)
    agg_chase = cross_dataset_eval(model, cfg, device, cfg.run_dir(), target="CHASEDB1")
    gap = {k: float(agg_drive.get(k, np.nan) - agg_chase.get(k, np.nan))
           for k in ["sensitivity", "specificity", "f1", "auc_roc", "accuracy"]}
    rows = [
        _row("DRIVE (in-dist)", agg_drive, {"target": "DRIVE-test"}),
        _row("CHASE (cross)",   agg_chase, {"target": "CHASEDB1"}),
        _row("gap (DRIVE-CHASE)", gap,    {"target": "diferencia"}),
    ]
    _save_table(rows, out_dir / "cross_dataset.csv")
    return rows


# --------------------------------------------------------------------------- #
# (C) Adaptación de dominio mediante CLAHE
# --------------------------------------------------------------------------- #
def domain_adaptation_study(device, base: C.Config, out_dir: Path):
    """Compara baseline (sin CLAHE) vs CLAHE y reporta cuánto cierra la brecha."""
    results = {}
    for tag, pre in [("baseline_noprep", "none"), ("clahe", "clahe")]:
        cfg = _override(base, preprocess=pre, run_name=f"C_{tag}")
        model, agg_drive = train_run(cfg, device, make_failure=False)
        agg_chase = cross_dataset_eval(model, cfg, device, cfg.run_dir(),
                                       target="CHASEDB1")
        results[tag] = {"drive": agg_drive, "chase": agg_chase}

    rows = []
    for tag, r in results.items():
        rows.append(_row(f"{tag}|DRIVE",  r["drive"], {"variant": tag, "target": "DRIVE"}))
        rows.append(_row(f"{tag}|CHASE",  r["chase"], {"variant": tag, "target": "CHASE"}))
        gap = {k: float(r["drive"].get(k, np.nan) - r["chase"].get(k, np.nan))
               for k in ["sensitivity", "specificity", "f1", "auc_roc", "accuracy"]}
        rows.append(_row(f"{tag}|gap",    gap,        {"variant": tag, "target": "gap"}))
    _save_table(rows, out_dir / "domain_adaptation.csv")
    return rows
