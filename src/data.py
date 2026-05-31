"""
Carga de los tres datasets del enunciado y sus DataLoaders.

Estructura esperada (tras descomprimir):

  data/
    DRIVE/
      training/  images/  1st_manual/  mask/
      test/      images/  1st_manual/  mask/   (2nd_manual opcional)
    STARE/
      stare-images/      im####.ppm
      labels-ah/         im####.ah.ppm
      labels-vk/         im####.vk.ppm
    CHASEDB1/
      Image_##L.jpg / Image_##R.jpg
      Image_##L_1stHO.png  (anotador primario)
      Image_##L_2ndHO.png  (anotador secundario)

Convención unificada: cada muestra es un dict con claves
  'image' (ruta), 'mask' (ruta GT), 'fov' (ruta o None), 'name'.
"""
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import preprocess_image, compute_fov_mask


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        # PPM/GIF pueden no leerse con cv2 según build; fallback a PIL
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_binary(path: str) -> np.ndarray:
    """Lee una máscara binaria como uint8 {0,255}."""
    from PIL import Image
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# Escaneo de cada dataset
# --------------------------------------------------------------------------- #
def scan_drive(data_root: Path, split: str) -> List[Dict]:
    """split = 'training' o 'test'.

    Nota: en la distribución oficial de DRIVE (grand-challenge), la carpeta
    test/ NO incluye 1st_manual/ (las etiquetas de test estaban ocultas para
    el challenge). Esta función devuelve lista vacía en ese caso y dejamos que
    get_drive_loaders haga una partición 3-vías sobre 'training' para tener
    un conjunto de test con GT.
    """
    base = Path(data_root) / "DRIVE" / split
    if not base.exists():
        return []
    imgs = sorted((base / "images").glob("*.tif")) + \
           sorted((base / "images").glob("*.tiff"))
    manual_dir = base / "1st_manual"
    if not manual_dir.exists():
        return []  # sin etiquetas de GT, no podemos evaluar este split
    samples = []
    for ip in imgs:
        stem = ip.stem  # ej. "21_training" o "01_test"
        idx = stem.split("_")[0]
        manual = manual_dir / f"{idx}_manual1.gif"
        mask = base / "mask" / f"{stem}_mask.gif"
        if not manual.exists() or not mask.exists():
            continue
        samples.append({
            "image": str(ip),
            "mask": str(manual),
            "fov": str(mask),
            "name": f"DRIVE_{split}_{idx}",
            "dataset": "DRIVE",
        })
    return samples


def scan_drive_inference_only(data_root: Path) -> List[Dict]:
    """DRIVE-test sin GT: solo imágenes (+ FOV si existe). Útil para inferencia."""
    base = Path(data_root) / "DRIVE" / "test"
    if not base.exists():
        return []
    imgs = sorted((base / "images").glob("*.tif")) + \
           sorted((base / "images").glob("*.tiff"))
    samples = []
    for ip in imgs:
        stem = ip.stem
        mask = base / "mask" / f"{stem}_mask.gif"
        samples.append({
            "image": str(ip),
            "mask": None,
            "fov": str(mask) if mask.exists() else None,
            "name": f"DRIVE_test_{stem.split('_')[0]}",
            "dataset": "DRIVE",
        })
    return samples


def scan_stare(data_root: Path) -> List[Dict]:
    base = Path(data_root) / "STARE"
    if not base.exists():
        return []
    img_dir = base / "stare-images"
    ah_dir = base / "labels-ah"
    vk_dir = base / "labels-vk"
    samples = []
    for ip in sorted(img_dir.glob("im*.ppm")):
        stem = ip.stem  # ej. "im0001"
        ah = ah_dir / f"{stem}.ah.ppm"
        vk = vk_dir / f"{stem}.vk.ppm"
        if not ah.exists():
            continue
        samples.append({
            "image": str(ip),
            "mask": str(ah),
            "mask_secondary": str(vk) if vk.exists() else None,
            "fov": None,
            "name": f"STARE_{stem}",
            "dataset": "STARE",
        })
    return samples


def scan_chasedb1(data_root: Path) -> List[Dict]:
    base = Path(data_root) / "CHASEDB1"
    if not base.exists():
        return []
    samples = []
    for ip in sorted(base.glob("Image_*.jpg")):
        stem = ip.stem
        primary = base / f"{stem}_1stHO.png"
        secondary = base / f"{stem}_2ndHO.png"
        if not primary.exists():
            continue
        samples.append({
            "image": str(ip),
            "mask": str(primary),
            "mask_secondary": str(secondary) if secondary.exists() else None,
            "fov": None,
            "name": f"CHASE_{stem}",
            "dataset": "CHASEDB1",
        })
    return samples


# --------------------------------------------------------------------------- #
# Augmentations sincronizadas (imagen + mask + fov)
# --------------------------------------------------------------------------- #
def _to_tensor_triplet(img: np.ndarray, mask: np.ndarray, fov: np.ndarray):
    """img uint8 [H,W,3] -> float[3,H,W] en [0,1]. mask/fov uint8 -> float[1,H,W] en {0,1}."""
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
    fov_t = torch.from_numpy(fov.astype(np.float32) / 255.0).unsqueeze(0)
    return img_t, mask_t, fov_t


def _augment(img_t: torch.Tensor, mask_t: torch.Tensor, fov_t: torch.Tensor,
             strength: str):
    """Aumento geométrico sincronizado + jitter de intensidad sobre la imagen."""
    if strength == "none":
        return img_t, mask_t, fov_t

    # Flips
    if random.random() < 0.5:
        img_t = TF.hflip(img_t); mask_t = TF.hflip(mask_t); fov_t = TF.hflip(fov_t)
    if random.random() < 0.5:
        img_t = TF.vflip(img_t); mask_t = TF.vflip(mask_t); fov_t = TF.vflip(fov_t)

    # Rotación
    max_angle = 30 if strength == "strong" else 15
    angle = random.uniform(-max_angle, max_angle)
    img_t = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask_t = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.NEAREST)
    fov_t = TF.rotate(fov_t, angle, interpolation=TF.InterpolationMode.NEAREST)

    # Jitter de intensidad (solo imagen)
    if random.random() < 0.7:
        b = random.uniform(0.8, 1.2); c = random.uniform(0.8, 1.2)
        img_t = TF.adjust_brightness(img_t, b)
        img_t = TF.adjust_contrast(img_t, c)
    if strength == "strong" and random.random() < 0.3:
        gamma = random.uniform(0.7, 1.4)
        img_t = TF.adjust_gamma(img_t, gamma)

    img_t = torch.clamp(img_t, 0.0, 1.0)
    return img_t, mask_t, fov_t


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FundusVesselDataset(Dataset):
    """
    Devuelve (image[3,H,W], mask[1,H,W], fov[1,H,W], name).
    Aplica preprocesamiento (none/green/clahe), redimensiona a img_size,
    y opcionalmente aumenta.
    """
    def __init__(self, samples: List[Dict], img_size: int = 512,
                 preprocess: str = "clahe", aug_strength: str = "none"):
        self.samples = samples
        self.img_size = img_size
        self.preprocess = preprocess
        self.aug_strength = aug_strength

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = _read_rgb(s["image"])
        mask = _read_binary(s["mask"])
        if s.get("fov"):
            fov = _read_binary(s["fov"])
        else:
            fov = compute_fov_mask(img)

        img = preprocess_image(img, self.preprocess)

        # Redimensionado uniforme
        size = (self.img_size, self.img_size)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        fov = cv2.resize(fov, size, interpolation=cv2.INTER_NEAREST)

        img_t, mask_t, fov_t = _to_tensor_triplet(img, mask, fov)
        img_t, mask_t, fov_t = _augment(img_t, mask_t, fov_t, self.aug_strength)
        # Binariza tras la rotación bilineal (no aplica a mask, pero por seguridad)
        mask_t = (mask_t > 0.5).float()
        fov_t = (fov_t > 0.5).float()
        return img_t, mask_t, fov_t, s["name"]


# --------------------------------------------------------------------------- #
# Constructores de DataLoaders
# --------------------------------------------------------------------------- #
def get_drive_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Loaders de entrenamiento/validación/test para DRIVE.

    Si DRIVE/test/1st_manual/ existe (versión redistribuida con etiquetas),
    usamos la partición fija del benchmark: train = DRIVE-training, test =
    DRIVE-test, y reservamos `val_fraction` de training para validación.

    Si NO existe (distribución oficial de grand-challenge), hacemos una
    partición 3-vías sobre DRIVE-training con la misma `val_fraction` aplicada
    a train y a test (por defecto: 60/20/20 sobre los 20 imágenes etiquetadas).
    """
    set_seed(cfg.seed)
    train_pool = scan_drive(cfg.data_root, "training")
    test_set = scan_drive(cfg.data_root, "test")  # vacío si no hay 1st_manual

    if not train_pool:
        raise FileNotFoundError(
            f"No se encontró DRIVE-training en {cfg.data_root}/DRIVE/training. "
            "Revisa el README para la descarga y la estructura esperada."
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(train_pool)

    if test_set:
        # Caso 1: el test tiene GT → partición fija del benchmark
        n_val = max(1, int(round(len(train_pool) * cfg.val_fraction)))
        val_set = train_pool[:n_val]
        train_set = train_pool[n_val:]
        split_kind = "benchmark_fijo"
    else:
        # Caso 2: el test no tiene GT (oficial). Partimos training en 3.
        n = len(train_pool)
        n_val = max(1, int(round(n * cfg.val_fraction)))
        n_test = max(1, int(round(n * cfg.val_fraction)))
        val_set = train_pool[:n_val]
        test_set = train_pool[n_val:n_val + n_test]
        train_set = train_pool[n_val + n_test:]
        split_kind = "3vias_sobre_training"
        print("[data] DRIVE/test no tiene 1st_manual; usando partición 3-vías "
              f"sobre DRIVE-training (train={len(train_set)}, val={len(val_set)}, "
              f"test={len(test_set)}).")

    train_ds = FundusVesselDataset(train_set, cfg.img_size, cfg.preprocess, cfg.aug_strength)
    val_ds = FundusVesselDataset(val_set, cfg.img_size, cfg.preprocess, aug_strength="none")
    test_ds = FundusVesselDataset(test_set, cfg.img_size, cfg.preprocess, aug_strength="none")

    common = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **common)

    info = {
        "split_kind": split_kind,
        "n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set),
    }
    return train_loader, val_loader, test_loader, info


def get_eval_loader(cfg, dataset_name: str) -> Tuple[DataLoader, Dict]:
    """Loader de evaluación para DRIVE-test, CHASEDB1 o STARE."""
    dataset_name = dataset_name.upper()
    if dataset_name == "DRIVE":
        samples = scan_drive(cfg.data_root, "test")
    elif dataset_name == "CHASEDB1":
        samples = scan_chasedb1(cfg.data_root)
    elif dataset_name == "STARE":
        samples = scan_stare(cfg.data_root)
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")

    if not samples:
        raise FileNotFoundError(
            f"No se encontraron muestras para {dataset_name} en {cfg.data_root}."
        )

    ds = FundusVesselDataset(samples, cfg.img_size, cfg.preprocess, aug_strength="none")
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    return loader, {"n_samples": len(samples), "dataset": dataset_name}
