"""
U-Net implementada desde cero (sin recurrir a librerías de segmentación).

Arquitectura clásica de Ronneberger et al. (2015) adaptada:
  - Codificador: `depth` bloques DoubleConv + MaxPool2d
  - Cuello de botella: DoubleConv
  - Decodificador: `depth` bloques de upsample + concat con skip + DoubleConv
  - Cabeza: Conv 1x1 a `n_classes` logits

Variantes soportadas:
  - upsample bilineal vs ConvTranspose
  - profundidad configurable (3 ó 4 etapas)
  - bloque "no-skip" (para ablación de las conexiones de salto)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Bloques
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d -> DoubleConv"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample + concat con skip + DoubleConv."""

    def __init__(self, in_c_below: int, skip_c: int, out_c: int,
                 bilinear: bool = True, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_c = in_c_below
        else:
            self.up = nn.ConvTranspose2d(in_c_below, in_c_below // 2,
                                         kernel_size=2, stride=2)
            up_c = in_c_below // 2
        in_conv = up_c + (skip_c if use_skip else 0)
        self.conv = DoubleConv(in_conv, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_skip:
            # corregir desajustes de 1px por redondeo
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    """
    U-Net con profundidad configurable.

    Args:
        in_channels: canales de entrada (3).
        n_classes:   número de logits de salida (1 para binario).
        base:        canales del primer nivel.
        depth:       número de etapas de bajada (=skip connections).
        bilinear:    si True usa Upsample bilineal; si False, ConvTranspose2d.
        use_skip:    activa/desactiva las conexiones de salto (para ablación).
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1,
                 base: int = 32, depth: int = 4,
                 bilinear: bool = True, use_skip: bool = True):
        super().__init__()
        assert depth >= 2, "depth debe ser >= 2"
        self.depth = depth

        # Canales por nivel: base, base*2, ..., base*2^depth
        chans = [base * (2 ** i) for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList([Down(chans[i], chans[i + 1]) for i in range(depth)])

        ups = []
        for i in range(depth, 0, -1):
            # entrada desde abajo: chans[i] ; skip que viene del nivel i-1: chans[i-1]
            ups.append(Up(in_c_below=chans[i], skip_c=chans[i - 1],
                          out_c=chans[i - 1], bilinear=bilinear, use_skip=use_skip))
        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(chans[0], n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.inc(x); skips.append(x)
        for i in range(self.depth - 1):
            x = self.downs[i](x); skips.append(x)
        x = self.downs[-1](x)  # cuello de botella
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 1)]
            x = up(x, skip)
        return self.outc(x)


def build_model(cfg) -> nn.Module:
    return UNet(
        in_channels=3, n_classes=1,
        base=cfg.base_channels, depth=cfg.depth,
        bilinear=cfg.bilinear, use_skip=getattr(cfg, "use_skip", True),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
