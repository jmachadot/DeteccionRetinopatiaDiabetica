"""
Funciones de pérdida para segmentación binaria píxel a píxel.
Todas aceptan una máscara FOV opcional para evaluar/optimizar solo donde
la imagen aporta información.

  - FovBCELoss: entropía cruzada binaria con logits, ponderada por FOV.
  - DiceLoss:   1 - dice = 1 - 2|X∩Y|/(|X|+|Y|).
  - CombinedLoss: w·BCE + (1-w)·Dice. Recomendada por defecto.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FovBCELoss(nn.Module):
    def forward(self, logits, target, fov=None):
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if fov is not None:
            denom = fov.sum().clamp(min=1.0)
            return (loss * fov).sum() / denom
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target, fov=None):
        prob = torch.sigmoid(logits)
        if fov is not None:
            prob = prob * fov
            target = target * fov
        dims = (1, 2, 3)  # canal + espaciales
        inter = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = FovBCELoss()
        self.dice = DiceLoss()
        self.w = bce_weight

    def forward(self, logits, target, fov=None):
        return self.w * self.bce(logits, target, fov) + \
               (1.0 - self.w) * self.dice(logits, target, fov)


def build_loss(cfg):
    name = cfg.loss.lower()
    if name == "bce":
        return FovBCELoss()
    if name == "dice":
        return DiceLoss()
    if name == "combined":
        return CombinedLoss(bce_weight=cfg.bce_weight)
    raise ValueError(f"Pérdida desconocida: {cfg.loss}")
