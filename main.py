"""
Punto de entrada de extremo a extremo.

  python main.py                # Pipeline completo: ablación + cross-dataset + CLAHE
  python main.py --quick        # 3 épocas; smoke test del pipeline
  python main.py --stage ablation    # solo ablación
  python main.py --stage cross       # solo generalización DRIVE -> CHASE
  python main.py --stage adaptation  # solo estudio de CLAHE
"""
import argparse
import json
from pathlib import Path

import torch

import config as C
import experiments as E


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(C.PROJECT_ROOT / "data"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="3 épocas, smoke test")
    p.add_argument("--stage", choices=["all", "ablation", "cross", "adaptation"],
                   default="all")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: no se detectó GPU; el entrenamiento será lento.")

    base = C.Config(
        data_root=args.data_root,
        epochs=3 if args.quick else args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    out_dir = Path(base.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.stage in ("all", "ablation"):
        results["ablation"] = E.ablation_study(device, base, out_dir)
    if args.stage in ("all", "cross"):
        results["cross_dataset"] = E.cross_dataset_study(device, base, out_dir)
    if args.stage in ("all", "adaptation"):
        results["domain_adaptation"] = E.domain_adaptation_study(device, base, out_dir)

    # Resumen Markdown
    lines = ["# Resumen de resultados\n"]
    for section, rows in results.items():
        lines.append(f"\n## {section}\n")
        if rows:
            keys = list(rows[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "|".join(["---"] * len(keys)) + "|")
            for r in rows:
                lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "all_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nListo. Resumen en {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
