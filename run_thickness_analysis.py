"""
Genera los archivos de análisis por grosor de vaso que las funciones de
experimentos omitieron para ahorrar tiempo:

  outputs/<run_name>/sensitivity_by_thickness.png
  outputs/<run_name>/sensitivity_by_thickness.json

Uso (desde la raíz del proyecto, con el venv activado):

    python run_thickness_analysis.py
    python run_thickness_analysis.py --run-name C_clahe
    python run_thickness_analysis.py --run-name C_baseline_noprep

Necesita que existan, en outputs/<run_name>/:
    config.json    (configuración con la que se entrenó)
    best_model.pt  (pesos del mejor modelo)

Y que esté disponible el dataset DRIVE en data/DRIVE/ (igual que cuando entrenaste).
"""
import argparse
import json
from pathlib import Path

import torch

import config as C
from src.data import get_drive_loaders
from src.models import build_model
from src.engine import evaluate
from src.failure_analysis import (
    sensitivity_by_thickness, plot_sensitivity_by_thickness,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="C_clahe",
                   help="Carpeta dentro de outputs/ con config.json y best_model.pt")
    p.add_argument("--output-dir", default=None,
                   help="Carpeta padre de los runs (por defecto, outputs/ del proyecto)")
    args = p.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else Path(C.PROJECT_ROOT) / "outputs"
    run_dir = out_root / args.run_name
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "best_model.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró {ckpt_path}")

    cfg = C.Config.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | run: {cfg.run_name} | preprocess: {cfg.preprocess}")

    # Reconstruimos los loaders (mismo split gracias a la semilla del config)
    _, _, test_loader, info = get_drive_loaders(cfg)
    print(f"[data] {info}")

    # Cargamos el modelo entrenado
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluamos para recuperar probabilidades, GT y máscaras FOV
    agg, _, dump = evaluate(model, test_loader, device,
                            threshold=cfg.threshold, tta=cfg.tta)
    print(f"[test] sens={agg['sensitivity']:.4f}  spec={agg['specificity']:.4f}  "
          f"f1={agg['f1']:.4f}  auc={agg['auc_roc']:.4f}")

    # Análisis por grosor de vaso
    by_thick = sensitivity_by_thickness(dump["probs"], dump["gts"], dump["fovs"],
                                        threshold=cfg.threshold)
    (run_dir / "sensitivity_by_thickness.json").write_text(
        json.dumps(by_thick, indent=2), encoding="utf-8"
    )
    plot_sensitivity_by_thickness(
        by_thick, run_dir / "sensitivity_by_thickness.png",
        title=f"Sensibilidad por grosor — {cfg.run_name}",
    )

    print(f"\nArchivos generados:")
    print(f"  {run_dir / 'sensitivity_by_thickness.json'}")
    print(f"  {run_dir / 'sensitivity_by_thickness.png'}")
    print(f"\nResumen por banda:")
    for band, m in by_thick.items():
        sens = m["sensitivity"]
        n = m["n_pixels"]
        rng = m["thickness_range_px"]
        print(f"  {band}: sens={sens:.4f} | {n:>8,} píxeles | rango {rng[0]:.1f}-{rng[1]} px")


if __name__ == "__main__":
    main()
