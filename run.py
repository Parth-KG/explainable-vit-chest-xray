"""
Command-line entry point. Examples:

    python run.py train          # fine-tune + save best checkpoint
    python run.py classify       # test-set accuracy/precision/recall/F1
    python run.py explain        # saliency maps for sample images
    python run.py xai-eval       # Insertion/Deletion, Entropy, AOPC table
    python run.py bonus          # baseline ranking-stability experiment
    python run.py all            # everything in order

Point Config.data_root at the COVID-19 Radiography Database first (README).
Use --synthetic to dry-run any stage without the dataset.
"""
import argparse, json, os

import numpy as np
import torch

from config import Config
from src import data, model as model_mod, train as train_mod
from src import explain, evaluate_xai, visualize, bonus


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("saved", path)


def get_loaders(cfg, synthetic):
    return (data.make_synthetic_loaders(cfg) if synthetic
            else data.make_loaders(cfg))


def cmd_train(cfg, synthetic):
    loaders, meta = get_loaders(cfg, synthetic)
    net = model_mod.build_model(cfg)
    res = train_mod.train(net, loaders, cfg, class_weights=meta.get("class_weights"))
    model_mod.save_checkpoint(net, cfg, extra=res)
    _save_json(res, f"{cfg.out_dir}/train_history.json")
    return net, loaders


def cmd_classify(cfg, net, loaders):
    m = train_mod.evaluate(net, loaders["test"], cfg)
    print(m["report"])
    visualize.save_confusion_matrix(m["confusion_matrix"], cfg.classes,
                                    f"{cfg.out_dir}/confusion_matrix.png")
    _save_json({k: v for k, v in m.items() if k != "report"},
               f"{cfg.out_dir}/classification_metrics.json")
    return m


def _sample_images(loaders, cfg, n=12):
    xs, ts = [], []
    for xb, yb in loaders["test"]:
        for i in range(xb.size(0)):
            xs.append(xb[i:i+1].to(cfg.device))
            if len(xs) >= n:
                break
        if len(xs) >= n:
            break
    net_targets = None
    return xs


def cmd_explain(cfg, net, loaders):
    xs = _sample_images(loaders, cfg, n=6)
    for idx, x in enumerate(xs):
        t = int(net(x).argmax(1).item())
        maps = explain.compute_all(net, x, t)
        visualize.save_method_comparison(
            x, maps, cfg.classes[t], f"{cfg.out_dir}/saliency/sample_{idx}.png")
    print(f"saved {len(xs)} saliency comparison figures to {cfg.out_dir}/saliency/")


def cmd_xai_eval(cfg, net, loaders, n_images=20):
    xs = _sample_images(loaders, cfg, n=n_images)
    method_names = list(explain.METHODS.keys())
    agg = {m: {k: [] for k in ("insertion_auc", "deletion_auc", "entropy", "aopc")}
           for m in method_names}
    curves_example = None
    for j, x in enumerate(xs):
        t = int(net(x).argmax(1).item())
        maps = explain.compute_all(net, x, t)
        per_method_curves = {}
        for m in method_names:
            r = evaluate_xai.evaluate_map(net, x, maps[m], t, cfg)
            for k in agg[m]:
                agg[m][k].append(r[k])
            per_method_curves[m] = r["_curves"]
        if j == 0:
            curves_example = per_method_curves
    table = {m: {k: float(np.mean(v)) for k, v in d.items()}
             for m, d in agg.items()}
    if curves_example:
        visualize.save_curves(curves_example, f"{cfg.out_dir}/insertion_deletion_curves.png")
    print("\n=== XAI evaluation (mean over %d images, baseline=%s) ===" %
          (len(xs), cfg.perturb_baseline))
    print(f"{'method':22s}{'ins_auc':>9}{'del_auc':>9}{'entropy':>9}{'aopc':>9}")
    for m, d in table.items():
        print(f"{m:22s}{d['insertion_auc']:9.3f}{d['deletion_auc']:9.3f}"
              f"{d['entropy']:9.3f}{d['aopc']:9.3f}")
    _save_json(table, f"{cfg.out_dir}/xai_metrics.json")
    return table


def cmd_bonus(cfg, net, loaders, n_images=20):
    xs = _sample_images(loaders, cfg, n=n_images)
    targets = [int(net(x).argmax(1).item()) for x in xs]
    res = bonus.rank_stability(net, xs, targets, cfg)
    print("\n=== BONUS: ranking stability across baselines ===")
    print("mean AOPC per baseline/method:")
    print(json.dumps(res["mean_aopc"], indent=2))
    print("rankings:", json.dumps(res["rankings"], indent=2))
    print("Kendall tau:", json.dumps(res["kendall_tau"], indent=2))
    print("mean tau:", round(res["mean_tau"], 3))
    _save_json(res, f"{cfg.out_dir}/bonus_rank_stability.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "classify", "explain",
                                      "xai-eval", "bonus", "all"])
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    synthetic = args.synthetic

    net, loaders = None, None
    if args.stage in ("train", "all"):
        net, loaders = cmd_train(cfg, synthetic)
    else:
        loaders, _ = get_loaders(cfg, synthetic)
        net = (model_mod.load_checkpoint(cfg)
               if os.path.exists(cfg.ckpt_path) else model_mod.build_model(cfg))

    if args.stage in ("classify", "all"):
        cmd_classify(cfg, net, loaders)
    if args.stage in ("explain", "all"):
        cmd_explain(cfg, net, loaders)
    if args.stage in ("xai-eval", "all"):
        cmd_xai_eval(cfg, net, loaders)
    if args.stage in ("bonus", "all"):
        cmd_bonus(cfg, net, loaders)


if __name__ == "__main__":
    main()
