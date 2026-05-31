"""
Inferencia con Test-Time Augmentation (TTA).

Promedia las predicciones de probabilidad sobre 4 versiones geométricas de cada
imagen: original, hflip, vflip y rotación 180°. Estrategia barata que mejora la
robustez del modelo en imágenes de dominio distinto al de entrenamiento.
"""
import torch


@torch.no_grad()
def predict_tta(model, img: torch.Tensor, device) -> torch.Tensor:
    """img: [1,3,H,W] en GPU. Devuelve probabilidad [1,1,H,W] promediada."""
    model.eval()
    img = img.to(device)
    probs = []

    def fwd(x):
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            return torch.sigmoid(model(x))

    # original
    probs.append(fwd(img))
    # flip horizontal
    probs.append(torch.flip(fwd(torch.flip(img, dims=[-1])), dims=[-1]))
    # flip vertical
    probs.append(torch.flip(fwd(torch.flip(img, dims=[-2])), dims=[-2]))
    # rotación 180° (= hflip + vflip)
    probs.append(torch.flip(fwd(torch.flip(img, dims=[-1, -2])), dims=[-1, -2]))

    return torch.stack(probs, dim=0).mean(dim=0)
