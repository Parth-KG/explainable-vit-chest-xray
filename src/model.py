"""Model factory built on timm. Default is ViT-B/16; any timm name works."""
import torch
import torch.nn as nn
import timm

from config import Config, IMAGENET_MEAN, IMAGENET_STD


def build_model(cfg: Config) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )
    return model.to(cfg.device)


def save_checkpoint(model: nn.Module, cfg: Config, extra: dict = None):
    import os
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_name": cfg.model_name,
                "classes": cfg.classes,
                "extra": extra or {}}, cfg.ckpt_path)


def load_checkpoint(cfg: Config) -> nn.Module:
    model = build_model(cfg)
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation -> tensor in [0,1] for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
