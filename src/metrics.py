"""
Métricas de segmentación binaria píxel a píxel.

Todas las métricas se calculan SOLO dentro de la máscara FOV, como es estándar
en los benchmarks DRIVE / STARE / CHASE_DB1. Se reportan promediadas por imagen
(macro), evitando que las imágenes grandes dominen.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-8


def metrics_one_image(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray,
                      threshold: float = 0.5) -> dict:
    """prob/gt/fov: arrays 2D float {0..1}. fov define los píxeles válidos."""
    fov_b = fov.astype(bool)
    p = prob[fov_b].astype(np.float64)
    y = (gt[fov_b] > 0.5).astype(np.uint8)
    pred = (p >= threshold).astype(np.uint8)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    sens = tp / (tp + fn + EPS)           # = recall
    spec = tn / (tn + fp + EPS)
    prec = tp / (tp + fp + EPS)
    acc = (tp + tn) / (tp + tn + fp + fn + EPS)
    f1 = 2 * tp / (2 * tp + fp + fn + EPS)  # = Dice

    try:
        auc = roc_auc_score(y, p) if (y.min() == 0 and y.max() == 1) else float("nan")
    except ValueError:
        auc = float("nan")

    return {"sensitivity": sens, "specificity": spec, "precision": prec,
            "accuracy": acc, "f1": f1, "auc_roc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def aggregate_metrics(per_image: list) -> dict:
    """Promedia métricas por imagen (excluye NaN del AUC). Suma TP/FP/FN/TN."""
    keys = ["sensitivity", "specificity", "precision", "accuracy", "f1", "auc_roc"]
    out = {}
    for k in keys:
        vals = [m[k] for m in per_image if not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    for k in ["tp", "fp", "fn", "tn"]:
        out[k] = int(sum(m[k] for m in per_image))
    return out


def format_metrics(m: dict) -> str:
    keys = ["sensitivity", "specificity", "f1", "auc_roc", "accuracy"]
    return " | ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #
def plot_roc_pixel(probs: np.ndarray, gts: np.ndarray, fovs: np.ndarray,
                   out_path: Path, title: str = "ROC píxel a píxel"):
    """ROC concatenando todos los píxeles dentro del FOV de todo el conjunto."""
    p_all, y_all = [], []
    for p, g, f in zip(probs, gts, fovs):
        m = f.astype(bool)
        p_all.append(p[m]); y_all.append((g[m] > 0.5).astype(np.uint8))
    p_all = np.concatenate(p_all); y_all = np.concatenate(y_all)
    fpr, tpr, _ = roc_curve(y_all, p_all)
    auc = roc_auc_score(y_all, p_all)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(auc)


def save_qualitative_grid(images, gts, preds, fovs, names, out_path: Path, n=4):
    """Cuadrícula imagen | GT | predicción | superposición de errores."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        img = images[r].transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        gt = gts[r]; pr = preds[r]; fv = fovs[r]
        # Errores: FN rojo, FP amarillo
        err = np.zeros((*gt.shape, 3), dtype=np.float32)
        err[..., 0] = ((pr == 0) & (gt == 1)).astype(np.float32)              # FN -> rojo
        err[..., 0] += ((pr == 1) & (gt == 0)).astype(np.float32) * 0.9       # FP -> rojo+verde
        err[..., 1] = ((pr == 1) & (gt == 0)).astype(np.float32) * 0.9
        overlay = 0.5 * img + 0.5 * err
        overlay = np.clip(overlay, 0, 1)

        for ax, im, title in zip(
            axes[r],
            [img, gt, pr, overlay],
            ["Imagen", "GT", "Predicción", "FN (rojo) / FP (amarillo)"],
        ):
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_title(f"{names[r]} — {title}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
