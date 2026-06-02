"""
Quantitative evaluation of saliency maps.

Implements the three metrics the assignment asks for:
  * Insertion / Deletion  (Petsiuk et al., RISE 2018)
  * Entropy               (focus / sparsity of the map)
  * AOPC                  (Samek et al., 2017 -- Area Over Perturbation Curve)

All perturbation-based metrics support a selectable baseline
("zero" | "blur" | "mean"); the choice is exactly what the bonus experiment
stress-tests, because off-manifold baselines (e.g. black pixels) are a known
source of unreliability.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def _make_baseline(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(x)
    if kind == "mean":
        return torch.ones_like(x) * x.mean(dim=(2, 3), keepdim=True)
    if kind == "blur":
        # large-kernel Gaussian blur: keeps low-frequency content, removes
        # the structure a saliency map points at -> stays closer to manifold
        k = 31
        coords = torch.arange(k, device=x.device) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * (k / 3) ** 2))
        g = (g / g.sum()).view(1, 1, k, 1)
        gx = g.transpose(2, 3)
        blurred = x
        for kern in (g, gx):
            kern = kern.expand(x.size(1), 1, *kern.shape[2:])
            blurred = F.conv2d(blurred, kern, padding="same", groups=x.size(1))
        return blurred
    raise ValueError(f"unknown baseline {kind}")


def _ordering(saliency: np.ndarray) -> np.ndarray:
    """Pixel indices sorted by importance, most-important first."""
    return np.argsort(saliency.ravel())[::-1]


@torch.no_grad()
def _prob_curve(model, x, baseline, order, target, n_steps, mode):
    """Step through the pixel ordering, blending image<->baseline, and record
    the target-class probability at each step.

    mode='deletion': start from image, replace top pixels with baseline.
    mode='insertion': start from baseline, reveal top pixels from image.
    """
    device = x.device
    H, W = x.shape[-2:]
    total = H * W
    step = max(1, total // n_steps)
    probs: List[float] = []
    fracs: List[float] = []

    for i in range(0, total + 1, step):
        mask = torch.zeros(total, device=device)
        mask[order[:i].copy()] = 1.0          # 1 = "use image" pixels selected
        mask = mask.view(1, 1, H, W)
        if mode == "deletion":
            img = x * (1 - mask) + baseline * mask
        else:  # insertion
            img = baseline * (1 - mask) + x * mask
        p = F.softmax(model(img), dim=1)[0, target].item()
        probs.append(p)
        fracs.append(i / total)
    return np.array(fracs), np.array(probs)


def insertion_deletion(model, x, saliency, target, n_steps=50,
                       baseline_kind="blur") -> Dict:
    baseline = _make_baseline(x, baseline_kind)
    order = _ordering(saliency)
    fd, pd = _prob_curve(model, x, baseline, order, target, n_steps, "deletion")
    fi, pi = _prob_curve(model, x, baseline, order, target, n_steps, "insertion")
    return {
        "deletion_auc": float(_trapz(pd, fd)),   # lower is better
        "insertion_auc": float(_trapz(pi, fi)),  # higher is better
        "deletion_curve": (fd.tolist(), pd.tolist()),
        "insertion_curve": (fi.tolist(), pi.tolist()),
    }


def entropy(saliency: np.ndarray) -> float:
    """Normalised Shannon entropy of the saliency map in [0,1].
    Lower = more focused/peaky; higher = more diffuse."""
    p = saliency.ravel().astype(np.float64)
    s = p.sum()
    if s <= 1e-12:
        return 1.0
    p = p / s
    nz = p[p > 0]
    h = -(nz * np.log(nz)).sum()
    return float(h / np.log(len(p)))   # normalise by max entropy


@torch.no_grad()
def aopc(model, x, saliency, target, n_steps=50, baseline_kind="blur") -> float:
    """Area Over the Perturbation Curve (MoRF -- Most Relevant First).
    Average drop in target probability as the most-relevant regions are
    progressively perturbed. Higher = more faithful explanation."""
    baseline = _make_baseline(x, baseline_kind)
    order = _ordering(saliency)
    H, W = x.shape[-2:]
    total = H * W
    step = max(1, total // n_steps)

    p0 = F.softmax(model(x), dim=1)[0, target].item()
    drops = []
    for i in range(step, total + 1, step):
        mask = torch.zeros(total, device=x.device)
        mask[order[:i].copy()] = 1.0
        mask = mask.view(1, 1, H, W)
        img = x * (1 - mask) + baseline * mask
        pk = F.softmax(model(img), dim=1)[0, target].item()
        drops.append(p0 - pk)
    return float(np.mean(drops))


def evaluate_map(model, x, saliency, target, cfg) -> Dict:
    """All three metrics for one (image, saliency) pair."""
    idd = insertion_deletion(model, x, saliency, target,
                             cfg.n_perturb_steps, cfg.perturb_baseline)
    return {
        "insertion_auc": idd["insertion_auc"],
        "deletion_auc": idd["deletion_auc"],
        "entropy": entropy(saliency),
        "aopc": aopc(model, x, saliency, target,
                     cfg.n_perturb_steps, cfg.perturb_baseline),
        "_curves": {"deletion": idd["deletion_curve"],
                    "insertion": idd["insertion_curve"]},
    }
