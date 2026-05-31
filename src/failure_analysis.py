"""
Análisis de fallos por grosor de vaso.

Cumple con el entregable 5 del enunciado: identificar qué tipo de vasos
(capilares finos vs arterias grandes) son más difíciles de segmentar y por qué.

Estrategia:
  1) Sobre la máscara ground-truth, aplicamos la transformada de distancia.
     Cada píxel positivo recibe su distancia al borde del vaso (≈ "radio local").
  2) Particionamos los píxeles positivos en 3 bandas según percentiles globales
     (finos: <p33, medios: p33-p66, gruesos: >p66).
  3) Calculamos sensibilidad por banda. Esperamos que los vasos finos tengan la
     sensibilidad más baja: son los más difíciles, ya que un par de capas de
     max-pool eliminan estructuras de pocos píxeles.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def vessel_thickness_map(gt_mask: np.ndarray) -> np.ndarray:
    """Transformada de distancia restringida al GT: aprox. del radio local."""
    gt = (gt_mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(gt, cv2.DIST_L2, 5)
    return dist


def sensitivity_by_thickness(probs, gts, fovs, threshold: float = 0.5,
                             percentiles=(33, 66)) -> dict:
    """Calcula sensibilidad por banda de grosor sobre todo el conjunto."""
    # 1) recolectar todos los valores de "grosor" sobre píxeles positivos
    all_dist = []
    for g, f in zip(gts, fovs):
        fb = f.astype(bool)
        d = vessel_thickness_map(g)
        all_dist.append(d[(g > 0.5) & fb])
    all_dist = np.concatenate(all_dist) if all_dist else np.array([0])

    p_lo, p_hi = np.percentile(all_dist, percentiles)
    bins = {"finos (<p33)": (0, p_lo),
            "medios (p33-p66)": (p_lo, p_hi),
            "gruesos (>p66)": (p_hi, np.inf)}

    counts = {k: {"tp": 0, "fn": 0} for k in bins}
    for p, g, f in zip(probs, gts, fovs):
        fb = f.astype(bool)
        d = vessel_thickness_map(g)
        pred = (p >= threshold)
        gt = (g > 0.5)
        for k, (lo, hi) in bins.items():
            sel = fb & gt & (d >= lo) & (d < hi)
            counts[k]["tp"] += int((pred & sel).sum())
            counts[k]["fn"] += int((~pred & sel).sum())

    out = {}
    for k, c in counts.items():
        tot = c["tp"] + c["fn"]
        out[k] = {
            "n_pixels": tot,
            "sensitivity": c["tp"] / tot if tot > 0 else float("nan"),
            "thickness_range_px": list(bins[k]),
        }
    return out


def plot_sensitivity_by_thickness(by_thick: dict, out_path: Path,
                                  title: str = "Sensibilidad por grosor de vaso"):
    keys = list(by_thick.keys())
    sens = [by_thick[k]["sensitivity"] for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(keys, sens, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sensibilidad")
    ax.set_title(title)
    for b, s in zip(bars, sens):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.02,
                f"{s:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
