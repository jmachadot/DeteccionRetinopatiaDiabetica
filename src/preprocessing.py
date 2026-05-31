"""
Preprocesamiento de imágenes de fondo de ojo:

  - Extracción del canal verde (donde los vasos contrastan más).
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) sobre el canal verde:
    aumenta el contraste local de los vasos y reduce diferencias entre dispositivos
    de captura — es nuestra estrategia principal de adaptación de dominio.
  - Cálculo automático del campo de visión (FOV) cuando el dataset no lo provee
    (STARE, CHASE_DB1).
"""
import cv2
import numpy as np


def green_channel_3ch(rgb_uint8: np.ndarray) -> np.ndarray:
    """Replica el canal verde a 3 canales (RGB->GGG)."""
    g = rgb_uint8[:, :, 1]
    return np.stack([g, g, g], axis=-1)


def apply_clahe_green(rgb_uint8: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size=(8, 8)) -> np.ndarray:
    """CLAHE sobre el canal verde, replicado a 3 canales."""
    g = rgb_uint8[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g_eq = clahe.apply(g)
    return np.stack([g_eq, g_eq, g_eq], axis=-1)


def preprocess_image(rgb_uint8: np.ndarray, mode: str) -> np.ndarray:
    """Devuelve uint8 [H,W,3] según el modo de preprocesamiento."""
    if mode == "none":
        return rgb_uint8
    if mode == "green":
        return green_channel_3ch(rgb_uint8)
    if mode == "clahe":
        return apply_clahe_green(rgb_uint8)
    raise ValueError(f"preprocess desconocido: {mode}")


def compute_fov_mask(rgb_uint8: np.ndarray, intensity_threshold: int = 15) -> np.ndarray:
    """
    Estima el campo de visión por umbralización del canal de luminancia y
    cierre morfológico. Útil cuando el dataset no provee FOV (STARE/CHASE).
    Devuelve mascara binaria uint8 {0,255}.
    """
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    _, fov = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
    # cerrar agujeros pequeños
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)
    # quedarse con el componente conexo más grande (el disco)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov = (labels == largest).astype(np.uint8) * 255
    return fov
