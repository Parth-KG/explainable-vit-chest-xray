"""
Central configuration for the Explainable Chest X-ray project.

Everything that you might want to tweak lives here so the rest of the code
stays clean. Edit DATA_ROOT to point at the COVID-19 Radiography Database
after you download it (see README).
"""
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    # ---- Data ----
    # Folder that contains the three class sub-folders. The COVID-19
    # Radiography Database ships as COVID/, Normal/, Viral Pneumonia/ each
    # holding an `images/` sub-folder; data.py handles both layouts.
    data_root: str = "data/COVID-19_Radiography_Dataset"
    # Classes are AUTO-DETECTED from the dataset folder at load time; this
    # list is just the default/fallback. Any folder whose name contains a
    # string in exclude_classes is dropped (keeps the task to 3 classes).
    classes: List[str] = field(
        default_factory=lambda: ["COVID", "Normal", "Viral Pneumonia"]
    )
    exclude_classes: List[str] = field(
        default_factory=lambda: ["lung_opacity", "lung opacity", "opacity", "masks"]
    )
    img_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

    # ---- Model ----
    # Any timm model works. ViT is the default because it gives us
    # Attention-Rollout for free. Swap to "resnet50" for a CNN baseline.
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True

    # ---- Training ----
    epochs: int = 8
    batch_size: int = 32
    lr: float = 3e-5            # small LR: we are fine-tuning, not training cold
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_workers: int = 4
    use_class_weights: bool = True   # dataset is imbalanced (Normal >> others)

    # ---- XAI evaluation ----
    # Number of perturbation steps for Insertion / Deletion / AOPC curves.
    n_perturb_steps: int = 50
    # Perturbation baseline used when "removing" pixels.
    # One of: "zero", "blur", "mean". The bonus experiment sweeps all three.
    perturb_baseline: str = "blur"

    # ---- Bookkeeping ----
    out_dir: str = "outputs"
    ckpt_path: str = "outputs/vit_xray_best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# ImageNet normalisation stats (timm pretrained models expect these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
