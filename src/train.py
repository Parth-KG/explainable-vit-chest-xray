"""Training + classification-metric reporting (accuracy, precision, recall, F1)."""
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from tqdm import tqdm

from config import Config


@torch.no_grad()
def _collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true, y_pred, classes) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(classes))),
        zero_division=0)
    return {
        "accuracy": float(acc),
        "macro": {"precision": float(p_macro), "recall": float(r_macro),
                  "f1": float(f1_macro)},
        "weighted": {"precision": float(p_w), "recall": float(r_w),
                     "f1": float(f1_w)},
        "per_class": {classes[i]: {"precision": float(p_c[i]),
                                   "recall": float(r_c[i]),
                                   "f1": float(f1_c[i])}
                      for i in range(len(classes))},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=classes, zero_division=0),
    }


def train(model, loaders, cfg: Config, class_weights=None) -> Dict:
    device = cfg.device
    weight = class_weights.to(device) if (class_weights is not None
                                          and cfg.use_class_weights) else None
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in tqdm(loaders["train"],
                           desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        y_true, y_pred = _collect_preds(model, loaders["val"], device)
        m = compute_metrics(y_true, y_pred, cfg.classes)
        history.append({"epoch": epoch + 1,
                        "train_loss": running / len(loaders["train"].dataset),
                        "val_accuracy": m["accuracy"],
                        "val_macro_f1": m["macro"]["f1"]})
        if m["macro"]["f1"] > best_f1:
            best_f1 = m["macro"]["f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_macro_f1": best_f1}


def evaluate(model, loader, cfg: Config) -> Dict:
    y_true, y_pred = _collect_preds(model, loader, cfg.device)
    return compute_metrics(y_true, y_pred, cfg.classes)
