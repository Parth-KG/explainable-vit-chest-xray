"""
Explainability methods. Each public function takes a model, a single
preprocessed image tensor x of shape [1,3,H,W], and a target class index,
and returns a saliency map as a numpy array of shape [H,W] normalised to
[0,1] (higher = more important for the predicted class).

Methods:
  * attention_rollout      -- ViT-specific, propagates attention across layers
  * integrated_gradients   -- model-agnostic attribution (Captum)
  * grad_cam               -- Grad-CAM adapted for ViT token grids
"""
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _normalize(sal: np.ndarray) -> np.ndarray:
    sal = sal.astype(np.float32)
    sal = sal - sal.min()
    rng = sal.max()
    if rng > 1e-12:
        sal = sal / rng
    return sal


def _to_hw(sal: torch.Tensor, size: int) -> np.ndarray:
    """Upsample a [h,w] or [1,1,h,w] tensor map to [size,size] and normalise."""
    if sal.dim() == 2:
        sal = sal[None, None]
    sal = F.interpolate(sal, size=(size, size), mode="bilinear",
                        align_corners=False)
    return _normalize(sal.squeeze().detach().cpu().numpy())


# --------------------------------------------------------------------------
# 1. Attention Rollout  (Abnar & Zuidema, 2020)
# --------------------------------------------------------------------------
class _AttnCatcher:
    """Disable fused attention on a timm ViT and capture per-block attention."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attentions = []
        for blk in model.blocks:
            blk.attn.fused_attn = False           # force explicit softmax path
            h = blk.attn.attn_drop.register_forward_hook(self._hook)
            self.handles.append(h)

    def _hook(self, module, inp, out):
        # input to attn_drop is the softmaxed attention: [B, heads, N, N]
        self.attentions.append(inp[0].detach())

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self.handles:
            h.remove()


def attention_rollout(model, x, target_class=None, head_fusion="mean",
                      discard_ratio=0.9) -> np.ndarray:
    """Propagate attention across all layers to get a CLS->patch saliency map.

    discard_ratio drops the lowest-weight attention edges at each layer to
    suppress noise (a standard trick from the original implementation)."""
    catcher = _AttnCatcher(model)
    catcher.clear()
    model.eval()
    with torch.no_grad():
        _ = model(x)
    attns = catcher.attentions          # list of [1, heads, N, N]
    catcher.remove()

    device = x.device
    N = attns[0].shape[-1]
    result = torch.eye(N, device=device)
    with torch.no_grad():
        for a in attns:
            if head_fusion == "mean":
                a = a.mean(dim=1)        # [1,N,N]
            elif head_fusion == "max":
                a = a.max(dim=1)[0]
            else:
                a = a.min(dim=1)[0]
            a = a[0]                     # [N,N]

            # discard lowest attentions (keep CLS column intact)
            flat = a.view(-1)
            n_keep = int(flat.numel() * (1 - discard_ratio))
            if 0 < n_keep < flat.numel():
                thresh = torch.topk(flat, n_keep, largest=True).values.min()
                a = torch.where(a >= thresh, a, torch.zeros_like(a))

            a = a + torch.eye(N, device=device)   # residual connection
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

    # CLS token (row 0) attention to the patch tokens (drop CLS col)
    mask = result[0, 1:]
    grid = int(mask.numel() ** 0.5)
    mask = mask.reshape(grid, grid)
    return _to_hw(mask, x.shape[-1])


# --------------------------------------------------------------------------
# 2. Integrated Gradients  (Sundararajan et al., 2017)
# --------------------------------------------------------------------------
def integrated_gradients(model, x, target_class, n_steps=32) -> np.ndarray:
    model.eval()
    if target_class is None:
        target_class = int(model(x).argmax(1).item())
    ig = IntegratedGradients(model)
    x = x.clone().requires_grad_(True)
    attr = ig.attribute(x, target=target_class, n_steps=n_steps,
                        baselines=x * 0)
    # collapse channels -> [H,W]; use absolute value (magnitude of influence)
    sal = attr.abs().sum(dim=1)          # [1,H,W]
    return _to_hw(sal[0], x.shape[-1])


# --------------------------------------------------------------------------
# 3. Grad-CAM for ViT  (Selvaraju et al., adapted with reshape_transform)
# --------------------------------------------------------------------------
def _vit_reshape_transform(tensor, height=14, width=14):
    # tensor: [B, N, C] with CLS token at position 0 -> drop it, reshape to grid
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)    # [B, C, H, W]


def grad_cam(model, x, target_class) -> np.ndarray:
    model.eval()
    if target_class is None:
        target_class = int(model(x).argmax(1).item())
    grid = x.shape[-1] // 16             # patch size 16 for vit_*_patch16
    reshape = lambda t: _vit_reshape_transform(t, grid, grid)
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape)
    sal = cam(input_tensor=x,
              targets=[ClassifierOutputTarget(target_class)])[0]   # [H,W]
    return _normalize(sal)


# Registry so callers can iterate cleanly.
METHODS: dict[str, Callable] = {
    "Attention-Rollout": attention_rollout,
    "Integrated-Gradients": integrated_gradients,
    "Grad-CAM": grad_cam,
}


def compute_all(model, x, target_class=None) -> dict[str, np.ndarray]:
    """Run every registered method on one image -> {name: [H,W] saliency}."""
    if target_class is None:
        target_class = int(model(x).argmax(1).item())
    out = {}
    for name, fn in METHODS.items():
        out[name] = fn(model, x, target_class)
    return out
