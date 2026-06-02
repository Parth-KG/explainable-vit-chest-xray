"""
BONUS CONTRIBUTION
==================
Research gap: perturbation-based saliency metrics (Deletion, Insertion, AOPC)
all "remove" pixels by replacing them with a baseline. The standard baseline
is zero (black). For chest X-rays this is doubly problematic: a black hole in
a radiograph is something the network has *never* seen, so the probability
drop it measures is partly a reaction to an off-manifold input rather than to
the loss of genuinely important evidence (cf. Hooker et al. ROAR 2019;
Rong et al. ROAD 2022).

Claim to validate: if a metric is trustworthy, the *ranking* of explanation
methods it produces should be stable under reasonable changes to the baseline.
We test this directly.

Experiment:
  * compute AOPC for every explanation method under {zero, blur, mean} baselines,
    averaged over a sample of test images;
  * rank the methods under each baseline;
  * measure ranking agreement with Kendall's tau between baseline pairs.

A low/unstable tau is evidence that the metric's verdict depends on an
arbitrary implementation choice -- a concrete, quantified limitation. The
'blur' baseline is proposed as the more manifold-respecting default because it
preserves low-frequency anatomy while removing the pointed-at structure.
"""
from itertools import combinations
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau

from src import explain, evaluate_xai


def rank_stability(model, images, targets, cfg,
                   baselines=("zero", "blur", "mean")) -> Dict:
    """images: list of [1,3,H,W] tensors; targets: list of int class ids."""
    method_names = list(explain.METHODS.keys())

    # aopc[baseline][method] = mean AOPC over the image sample
    aopc = {b: {m: [] for m in method_names} for b in baselines}
    for x, t in zip(images, targets):
        maps = explain.compute_all(model, x, t)
        for b in baselines:
            for m in method_names:
                aopc[b][m].append(
                    evaluate_xai.aopc(model, x, maps[m], t,
                                      cfg.n_perturb_steps, b))
    mean_aopc = {b: {m: float(np.mean(v)) for m, v in d.items()}
                 for b, d in aopc.items()}

    # rank methods (descending AOPC = best first) per baseline
    rankings = {b: [m for m, _ in sorted(mean_aopc[b].items(),
                                         key=lambda kv: kv[1], reverse=True)]
                for b in baselines}

    # Kendall tau between every pair of baselines
    taus = {}
    for b1, b2 in combinations(baselines, 2):
        r1 = [rankings[b1].index(m) for m in method_names]
        r2 = [rankings[b2].index(m) for m in method_names]
        tau, _ = kendalltau(r1, r2)
        taus[f"{b1}_vs_{b2}"] = float(tau)

    return {
        "mean_aopc": mean_aopc,
        "rankings": rankings,
        "kendall_tau": taus,
        "mean_tau": float(np.mean(list(taus.values()))) if taus else 1.0,
        "n_images": len(images),
    }
