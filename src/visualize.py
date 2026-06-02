"""Plotting helpers that save figures for the report."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import denormalize


def _img_for_show(x):
    img = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    return img


def overlay(img, saliency, alpha=0.5):
    import matplotlib.cm as cm
    heat = cm.jet(saliency)[..., :3]
    return (1 - alpha) * img + alpha * heat


def save_method_comparison(x, maps, pred_label, out_path):
    """One row: original + each method's heatmap overlay."""
    img = _img_for_show(x)
    n = len(maps) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    ax[0].imshow(img); ax[0].set_title(f"Input\n(pred: {pred_label})")
    ax[0].axis("off")
    for i, (name, sal) in enumerate(maps.items(), start=1):
        ax[i].imshow(overlay(img, sal)); ax[i].set_title(name)
        ax[i].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_curves(curves_by_method, out_path):
    """Deletion + insertion curves for every method on two panels."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, c in curves_by_method.items():
        fd, pd = c["deletion"]
        fi, pi = c["insertion"]
        ax[0].plot(fd, pd, label=name)
        ax[1].plot(fi, pi, label=name)
    ax[0].set_title("Deletion (lower AUC = better)")
    ax[0].set_xlabel("fraction removed"); ax[0].set_ylabel("target prob")
    ax[1].set_title("Insertion (higher AUC = better)")
    ax[1].set_xlabel("fraction inserted"); ax[1].set_ylabel("target prob")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm, classes, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
