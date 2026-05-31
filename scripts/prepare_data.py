"""
Verificación de la estructura de los datasets.

Los 3 datasets (DRIVE, STARE, CHASE_DB1) requieren registro y descarga manual:

  - DRIVE:    https://drive.grand-challenge.org/  (requiere cuenta)
  - STARE:    https://cecas.clemson.edu/~ahoover/stare/
  - CHASE_DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/

Una vez descargados, descomprime cada uno dentro de data/ siguiendo esta
estructura:

  data/
    DRIVE/
      training/  images/  1st_manual/  mask/
      test/      images/  1st_manual/  mask/
    STARE/
      stare-images/    labels-ah/    labels-vk/
    CHASEDB1/
      Image_01L.jpg ... + máscaras *_1stHO.png y *_2ndHO.png

Luego:
  python scripts/prepare_data.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from src.data import scan_drive, scan_stare, scan_chasedb1  # noqa: E402


def verify(data_root: Path):
    print(f"Verificando datasets en: {data_root}\n")

    drive_train = scan_drive(data_root, "training")
    drive_test = scan_drive(data_root, "test")
    stare = scan_stare(data_root)
    chase = scan_chasedb1(data_root)

    print(f"DRIVE training : {len(drive_train):>3} imágenes con GT  "
          f"({'OK' if len(drive_train) >= 18 else 'INCOMPLETO o ausente'})")
    if len(drive_test) > 0:
        print(f"DRIVE test     : {len(drive_test):>3} imágenes con GT  "
              "(versión redistribuida con etiquetas; se usará partición fija)")
    else:
        print(f"DRIVE test     :   0 imágenes con GT  "
              "(distribución oficial; se hará partición 3-vías de training)")
    print(f"STARE          : {len(stare):>3} imágenes  "
          f"({'OK' if len(stare) >= 18 else 'INCOMPLETO o ausente'})")
    print(f"CHASE_DB1      : {len(chase):>3} imágenes  "
          f"({'OK' if len(chase) >= 24 else 'INCOMPLETO o ausente'})")

    print("\nDRIVE-training es OBLIGATORIO (de allí sale el entrenamiento).")
    print("CHASE_DB1 es OBLIGATORIO (evaluación cruzada de dominio).")
    print("STARE y DRIVE-test con GT son opcionales.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(C.PROJECT_ROOT / "data"))
    args = p.parse_args()
    verify(Path(args.data_root))
