"""
Motor de entrenamiento y validación.

- Optimizador AdamW + scheduler cosine (configurable).
- Pérdida con FOV (BCE/Dice/combinada).
- Precisión mixta (AMP) en GPU.
- Early stopping y selección del mejor checkpoint por F1 de validación.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.metrics import metrics_one_image, aggregate_metrics, format_metrics


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, tta: bool = False):
    """Evalúa modelo sobre un loader. Devuelve métricas agregadas + listas por imagen.
    Si tta=True, promedia predicciones sobre 4 variantes geométricas."""
    from src.inference import predict_tta
    model.eval()
    per_image = []
    probs_list, gts_list, fovs_list, imgs_list, names_list = [], [], [], [], []
    for img, mask, fov, name in loader:
        img = img.to(device, non_blocking=True)
        if tta:
            prob_t = predict_tta(model, img, device)
        else:
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")): # type: ignore
                prob_t = torch.sigmoid(model(img))
        prob = prob_t.cpu().numpy()[0, 0]
        gt = mask.numpy()[0, 0]
        fv = fov.numpy()[0, 0]
        m = metrics_one_image(prob, gt, fv, threshold)
        per_image.append(m)
        probs_list.append(prob); gts_list.append(gt); fovs_list.append(fv)
        imgs_list.append(img.cpu().numpy()[0])
        names_list.append(name[0] if isinstance(name, (list, tuple)) else name)
    agg = aggregate_metrics(per_image)
    return agg, per_image, {
        "probs": probs_list, "gts": gts_list, "fovs": fovs_list,
        "imgs": imgs_list, "names": names_list,
    }


def _build_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=0.9, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    return None


def train_model(model, train_loader, val_loader, criterion, cfg, device,
                run_dir: Path = None, verbose=True):
    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda")) # type: ignore

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, disable=not verbose,
                    desc=f"Época {epoch}/{cfg.epochs}")
        for img, mask, fov, _ in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fov = fov.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")): # type: ignore
                logits = model(img)
                loss = criterion(logits, mask, fov)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            running += loss.item() * img.size(0); n += img.size(0)
            pbar.set_postfix(loss=running / max(n, 1))

        if scheduler is not None:
            scheduler.step()

        val_agg, _, _ = evaluate(model, val_loader, device, threshold=cfg.threshold)
        train_loss = running / max(n, 1)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in val_agg.items()
               if k in ("sensitivity", "specificity", "f1", "auc_roc", "accuracy")},
        })
        if verbose:
            print(f"  [val] loss_train={train_loss:.4f} | {format_metrics(val_agg)}")

        if val_agg["f1"] > best_f1:
            best_f1 = val_agg["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                if verbose:
                    print(f"  Early stopping en época {epoch} (mejor época: {best_epoch}).")
                break

    model.load_state_dict(best_state)
    if run_dir is not None:
        torch.save(best_state, run_dir / "best_model.pt")
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    return model, history, best_epoch
