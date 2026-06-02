"""
End-to-end smoke test on SYNTHETIC data (no dataset / GPU needed).

Runs the full pipeline -- build model -> train 1 epoch -> classification
metrics -> all 3 saliency methods -> all 3 evaluation metrics -> verifies
shapes and value ranges. If this passes, the code is wired correctly and the
only thing that changes for the real run is the data and the model size.

Run:  python -m tests.smoke_test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import Config
from src import data, model as model_mod, train as train_mod
from src import explain, evaluate_xai


def main():
    cfg = Config()
    # shrink everything for a fast CPU run
    cfg.model_name = "vit_tiny_patch16_224"
    cfg.pretrained = False
    cfg.epochs = 1
    cfg.batch_size = 8
    cfg.num_workers = 0
    cfg.n_perturb_steps = 8
    cfg.device = "cpu"

    print("1) synthetic loaders ...")
    loaders, meta = data.make_synthetic_loaders(cfg, n_per_class=16)

    print("2) build model ...")
    net = model_mod.build_model(cfg)

    print("3) train 1 epoch ...")
    train_mod.train(net, loaders, cfg, class_weights=meta["class_weights"])

    print("4) classification metrics ...")
    m = train_mod.evaluate(net, loaders["test"], cfg)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert set(m["per_class"]) == set(cfg.classes)
    print("   accuracy:", round(m["accuracy"], 3),
          "macro-F1:", round(m["macro"]["f1"], 3))

    print("5) saliency methods on one image ...")
    xb, yb = next(iter(loaders["test"]))
    x = xb[:1].to(cfg.device)
    target = int(net(x).argmax(1).item())
    maps = explain.compute_all(net, x, target)
    for name, sal in maps.items():
        assert sal.shape == (cfg.img_size, cfg.img_size), (name, sal.shape)
        assert 0.0 <= sal.min() and sal.max() <= 1.0 + 1e-5, name
        assert np.isfinite(sal).all(), name
        print(f"   {name:22s} shape={sal.shape} range=[{sal.min():.2f},{sal.max():.2f}]")

    print("6) XAI evaluation metrics ...")
    for name, sal in maps.items():
        res = evaluate_xai.evaluate_map(net, x, sal, target, cfg)
        for k in ("insertion_auc", "deletion_auc", "entropy", "aopc"):
            assert np.isfinite(res[k]), (name, k)
        print(f"   {name:22s} ins={res['insertion_auc']:.3f} "
              f"del={res['deletion_auc']:.3f} ent={res['entropy']:.3f} "
              f"aopc={res['aopc']:.3f}")

    print("7) baseline sweep (bonus) ...")
    for b in ("zero", "blur", "mean"):
        cfg.perturb_baseline = b
        r = evaluate_xai.aopc(net, x, maps["Grad-CAM"], target,
                              cfg.n_perturb_steps, b)
        print(f"   baseline={b:5s} AOPC={r:.3f}")

    print("\nSMOKE TEST PASSED \u2713")


if __name__ == "__main__":
    main()
