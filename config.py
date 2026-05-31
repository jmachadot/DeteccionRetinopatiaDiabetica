"""
Configuración central. Los scripts CLI sobreescriben cualquier campo.

Convención:
  - Clase positiva (1) = píxel de vaso sanguíneo
  - Clase negativa (0) = fondo
  - Todas las métricas se calculan SOLO dentro del campo de visión (FOV).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- Datos ---
    data_root: str = str(PROJECT_ROOT / "data")
    img_size: int = 512                  # debe ser múltiplo de 2^depth
    val_fraction: float = 0.2            # fracción de DRIVE-train reservada para validación
    num_workers: int = 4
    seed: int = 42

    # --- Modelo ---
    base_channels: int = 32
    depth: int = 4                       # nº de etapas de bajada (= skip connections)
    bilinear: bool = True                # upsample bilineal (vs ConvTranspose)

    # --- Entrenamiento ---
    epochs: int = 80
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"            # "cosine" | "step" | "none"
    early_stopping_patience: int = 15

    # --- Pérdida ---
    loss: str = "combined"               # "bce" | "dice" | "combined"
    bce_weight: float = 0.5              # peso de BCE dentro de la combinada

    # --- Preprocesamiento (clave para adaptación de dominio) ---
    preprocess: str = "clahe"            # "none" | "green" | "clahe"
    aug_strength: str = "standard"       # "none" | "standard" | "strong"

    # --- Inferencia ---
    tta: bool = False                    # test-time augmentation (flips + rot90)
    threshold: float = 0.5

    # --- Salidas ---
    output_dir: str = str(PROJECT_ROOT / "outputs")
    run_name: str = "default"

    def run_dir(self) -> Path:
        d = Path(self.output_dir) / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
