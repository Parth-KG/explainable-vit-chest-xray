"""
Data loading with AUTO-DETECTION of the dataset structure.

Works with whatever the provided dataset looks like:
  * flat layout:      root/<Class>/(images/)*.png        -> we make the split
  * pre-split layout: root/{train,val,test}/<Class>/*.png -> we use the split

Class names are discovered automatically; a small denylist drops the
Lung-Opacity category (the assignment is a 3-class task: COVID / Normal /
Viral Pneumonia). Edit Config.exclude_classes to change that.

`make_synthetic_loaders` is kept for the offline smoke test.
"""
import os
import glob
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T

from config import Config, IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_transforms(cfg: Config):
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=7),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class XrayDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def _images_in(folder: str) -> List[str]:
    """All images directly in `folder` or one level down (e.g. an images/ dir)."""
    out = [f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMG_EXT)]
    if not out:
        out = [f for f in glob.glob(os.path.join(folder, "*", "*")) if f.lower().endswith(IMG_EXT)]
    return out


def _is_excluded(name: str, cfg: Config) -> bool:
    n = name.lower().strip()
    return any(x in n for x in [e.lower() for e in cfg.exclude_classes])


def _find_dataset_root(root: str) -> str:
    """Descend through single-child wrapper folders to the real dataset root
    (the folder that directly contains the class sub-folders)."""
    cur = root
    for _ in range(5):
        subs = [d for d in glob.glob(os.path.join(cur, "*")) if os.path.isdir(d)]
        # if exactly one sub-dir and it has no images itself, descend
        if len(subs) == 1 and not _images_in(subs[0]) and \
           [d for d in glob.glob(os.path.join(subs[0], "*")) if os.path.isdir(d)]:
            cur = subs[0]
        else:
            break
    return cur


def _class_dirs(root: str, cfg: Config):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and _images_in(d) and not _is_excluded(os.path.basename(d), cfg))
    return dirs


def discover(root: str, cfg: Config) -> Dict:
    """Return a dict describing the dataset and the (path,label) items by split."""
    root = _find_dataset_root(root)
    top = [os.path.basename(d) for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    presplit = sum(1 for t in top if t.lower() in SPLIT_NAMES) >= 2

    if presplit:
        def split_dir(*names):
            for n in names:
                for t in top:
                    if t.lower() == n:
                        return os.path.join(root, t)
            return None
        train_d = split_dir("train", "training")
        test_d = split_dir("test", "testing")
        val_d = split_dir("val", "valid", "validation")
        # class list from train folder
        classes = [os.path.basename(d) for d in _class_dirs(train_d, cfg)]
        cidx = {c: i for i, c in enumerate(classes)}
        def items_from(d):
            out = []
            if d:
                for c in classes:
                    out += [(p, cidx[c]) for p in _images_in(os.path.join(d, c))]
            return out
        train = items_from(train_d); test = items_from(test_d); val = items_from(val_d)
        if not val:  # carve a val set out of train if none provided
            rng = random.Random(cfg.seed); rng.shuffle(train)
            n_val = int(len(train) * cfg.val_split)
            val, train = train[:n_val], train[n_val:]
        layout = "pre-split"
    else:
        cdirs = _class_dirs(root, cfg)
        classes = [os.path.basename(d) for d in cdirs]
        cidx = {c: i for i, c in enumerate(classes)}
        allitems = []
        for c in classes:
            allitems += [(p, cidx[c]) for p in _images_in(os.path.join(root, c))]
        # stratified split
        rng = random.Random(cfg.seed)
        by = {}
        for p, l in allitems:
            by.setdefault(l, []).append(p)
        train, val, test = [], [], []
        for l, ps in by.items():
            ps = sorted(ps); rng.shuffle(ps); n = len(ps)
            nt, nv = int(n*cfg.test_split), int(n*cfg.val_split)
            test += [(p, l) for p in ps[:nt]]
            val += [(p, l) for p in ps[nt:nt+nv]]
            train += [(p, l) for p in ps[nt+nv:]]
        rng.shuffle(train)
        layout = "flat (auto-split %d/%d/%d)" % (
            round((1-cfg.val_split-cfg.test_split)*100),
            round(cfg.val_split*100), round(cfg.test_split*100))

    return {"root": root, "classes": classes, "layout": layout,
            "train": train, "val": val, "test": test}


def make_loaders(cfg: Config):
    """Auto-detect structure, set cfg.classes, return loaders + meta."""
    _seed_everything(cfg.seed)
    info = discover(cfg.data_root, cfg)
    cfg.classes = info["classes"]            # <-- model/metrics use detected classes
    if not cfg.classes:
        raise FileNotFoundError(
            f"No class folders found under {cfg.data_root}. "
            f"Point Config.data_root at the dataset folder.")

    train_tf, eval_tf = build_transforms(cfg)
    dl = lambda items, tf, sh: DataLoader(
        XrayDataset(items, tf), batch_size=cfg.batch_size, shuffle=sh,
        num_workers=cfg.num_workers, pin_memory=True)
    loaders = {"train": dl(info["train"], train_tf, True),
               "val": dl(info["val"], eval_tf, False),
               "test": dl(info["test"], eval_tf, False)}

    counts = np.zeros(len(cfg.classes))
    for _, y in info["train"]:
        counts[y] += 1
    cw = (counts.sum() / (len(cfg.classes) * np.maximum(counts, 1))).astype(np.float32)

    # human-readable banner so the user can sanity-check before training
    print("=" * 64)
    print("DATASET DETECTED")
    print("  root   :", info["root"])
    print("  layout :", info["layout"])
    print("  classes:", cfg.classes)
    print("  images : train=%d  val=%d  test=%d" %
          (len(info["train"]), len(info["val"]), len(info["test"])))
    print("  per-class (train):",
          {cfg.classes[i]: int(counts[i]) for i in range(len(cfg.classes))})
    print("=" * 64)

    meta = {"class_weights": torch.tensor(cw),
            "n_train": len(info["train"]), "n_val": len(info["val"]),
            "n_test": len(info["test"]), "layout": info["layout"]}
    return loaders, meta


def make_synthetic_loaders(cfg: Config, n_per_class: int = 24):
    _seed_everything(cfg.seed)
    if not cfg.classes:
        cfg.classes = ["COVID", "Normal", "Viral Pneumonia"]
    k = len(cfg.classes); N = n_per_class * k
    x = torch.randn(N, 3, cfg.img_size, cfg.img_size) * 0.5
    y = torch.arange(k).repeat_interleave(n_per_class)
    for i in range(N):
        c = int(y[i]); r0 = 20 + c * 50
        x[i, :, r0:r0+30, r0:r0+30] += 2.0
    perm = torch.randperm(N); x, y = x[perm], y[perm]
    nt = int(N*cfg.test_split); nv = int(N*cfg.val_split)
    sp = {"test": TensorDataset(x[:nt], y[:nt]),
          "val": TensorDataset(x[nt:nt+nv], y[nt:nt+nv]),
          "train": TensorDataset(x[nt+nv:], y[nt+nv:])}
    loaders = {k_: DataLoader(v, batch_size=cfg.batch_size, shuffle=(k_=="train"))
               for k_, v in sp.items()}
    return loaders, {"class_weights": torch.ones(k), "synthetic": True}
