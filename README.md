# Explainable Vision Transformer for Chest X-ray Diagnosis

> Beyond a single heatmap: training a ViT to classify COVID-19 / Normal / Viral Pneumonia, then **quantitatively measuring whether its explanations can be trusted.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Model-ViT--B%2F16-F7931E">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20IG%20%7C%20Attention--Rollout-6f42c1">
  <img src="https://img.shields.io/badge/License-MIT-2ea44f">
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/<YOUR_USERNAME>/<YOUR_REPO>/blob/main/notebooks/colab_provided_dataset.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>

---

## Overview

This project was built for the **AIMS-DTU Research Intern 2026** selection round (Explainable Computer Vision track). It does two things most explainability demos skip:

1. **Trains a strong classifier** — a fine-tuned Vision Transformer (ViT-B/16) on a three-class chest X-ray task.
2. **Treats explainability as a measurable property, not a picture.** Three attribution methods are compared with three faithfulness metrics, and a bonus study asks the uncomfortable question: *do these metrics even agree with themselves?*

The headline finding is not the accuracy — it is that the three explanation methods **disagree on where the model looks**, the evaluation metrics **disagree on which explanation is best**, and faithfulness scores are **sensitive to an arbitrary implementation choice** (the perturbation baseline). In a clinical setting, that matters.

---

## Highlights

- **ViT-B/16** fine-tuned for COVID-19 / Normal / Viral Pneumonia classification.
- **Three explainability methods:** Attention-Rollout (ViT-native), Integrated Gradients, and Grad-CAM (adapted for transformers).
- **Three quantitative metrics:** Insertion/Deletion AUC, saliency Entropy, and AOPC (Area Over the Perturbation Curve).
- **Bonus research contribution:** a controlled study of how faithfulness scores shift under different perturbation baselines (zero / blur / mean), with ranking-stability measured by Kendall's tau.
- **Reproducible by design:** fixed seeds, an auto-detecting data loader (handles flat *or* pre-split datasets), a one-command pipeline, an end-to-end synthetic-data test, and a one-click Colab notebook.

---

## Results

> Held-out test set (435 images, balanced across the three classes).

| Metric | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|:------:|:--------:|:-----------------:|:--------------:|:----------:|
| **ViT-B/16** | **98.85%** | **98.87%** | **98.85%** | **98.85%** |

Per-class F1: COVID 98.95% · Normal 98.64% · Viral Pneumonia 98.96%. Only 5 of 435 test images are misclassified, with zero errors on the Normal class.

**Explainability comparison** (mean over 20 test images, blur baseline):

| Method | Insertion (up) | Deletion (down) | Entropy (down) | AOPC (up) |
|:-------|:-----------:|:----------:|:---------:|:------:|
| Attention-Rollout    | 0.895 | **0.872** | **0.937** | **0.041** |
| Integrated Gradients | **0.902** | 0.901 | 0.948 | 0.012 |
| Grad-CAM             | 0.897 | **0.872** | 0.962 | **0.041** |

Attention-Rollout and Grad-CAM are near-tied on faithfulness; Integrated Gradients produces sparse maps but the weakest AOPC. **Crucially, the ranking of methods is not stable across perturbation baselines** — see the bonus below.

---

## Methods at a glance

| Method | Type | What it answers |
|:-------|:-----|:----------------|
| **Attention-Rollout** | ViT-native | How attention propagates from input patches to the classification token across all layers |
| **Integrated Gradients** | Model-agnostic attribution | Per-pixel contribution, integrated along a path from a baseline to the input |
| **Grad-CAM** | Gradient-weighted activations | Coarse, class-discriminative region the final block responds to |

| Metric | Direction | Measures |
|:-------|:---------:|:---------|
| **Insertion / Deletion** | ins up / del down | Faithfulness — does revealing/removing important pixels change the prediction? |
| **Entropy** | down | Focus — is the saliency concentrated or diffuse? |
| **AOPC** | up | Faithfulness — average confidence drop when most-relevant regions are perturbed |

---

## Quickstart

### Option A — Google Colab (recommended)
Open the notebook badge above, set the runtime to **GPU**, upload the dataset, and run the cells top to bottom. The loader auto-detects the dataset structure and prints it for you to confirm before training.

### Option B — Local

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
pip install -r requirements.txt

# point config.py's data_root at your dataset folder, then:
python run.py all          # train -> classify -> explain -> xai-eval -> bonus
```

Individual stages:

```bash
python run.py train        # fine-tune ViT, save best checkpoint
python run.py classify     # accuracy / precision / recall / F1 + confusion matrix
python run.py explain      # saliency overlays for sample images
python run.py xai-eval     # Insertion/Deletion, Entropy, AOPC table
python run.py bonus        # baseline ranking-stability experiment
```

No dataset handy? Every stage runs on generated data with `--synthetic`, and the full pipeline is covered by:

```bash
python -m tests.smoke_test
```

---

## Dataset

A three-class chest X-ray dataset (COVID-19, Normal, Viral Pneumonia). The data loader **auto-detects** the layout:

- **Flat** — `root/<Class>/(images/)*.png` -> a stratified 70/15/15 split is created.
- **Pre-split** — `root/{train,val,test}/<Class>/*.png` -> the provided split is used.

It also skips `__MACOSX`/hidden folders and drops a Lung-Opacity category if present, keeping the task to three classes. Trained weights and full-resolution outputs are hosted on Drive (link below) rather than committed to the repo.

---

## Repository structure

```
config.py                       # all hyperparameters & paths
run.py                          # CLI orchestrator
src/
  data.py                       # auto-detecting loader, transforms, synthetic data
  model.py                      # timm model build/save/load
  train.py                      # training loop + classification metrics
  explain.py                    # Attention-Rollout, Integrated Gradients, Grad-CAM
  evaluate_xai.py               # Insertion/Deletion, Entropy, AOPC (+ baselines)
  bonus.py                      # baseline ranking-stability study (Kendall's tau)
  visualize.py                  # report figures
tests/smoke_test.py             # end-to-end correctness test
notebooks/                      # one-click Colab pipeline
```

---

## The bonus, in one paragraph

Perturbation-based metrics (Deletion, Insertion, AOPC) "remove" pixels by replacing them with a baseline — usually black. For an X-ray, a black hole is an image the model has never seen, so the measured confidence drop partly reflects an *off-manifold input* rather than lost evidence (cf. ROAR, ROAD). This project recomputes AOPC under zero / blur / mean baselines and reports ranking agreement via Kendall's tau. **The result is striking: the ranking of methods flips** — Grad-CAM is judged the *least* faithful under a black baseline but the *second-most* faithful under blur and mean, dropping Kendall's tau to 0.33 between baselines (mean tau approximately 0.56). A faithfulness score is therefore meaningful only *within* a fixed, manifold-respecting baseline — which is why this repo defaults to a Gaussian-blur baseline and reports stability alongside the scores.

---

## Limitations & future work

Near-perfect accuracy on aggregated COVID X-ray datasets can reflect source-specific artifacts rather than pathology, so results are reported with that caveat. Natural next steps: validate saliency against the dataset's lung masks (IoU), add the sanity checks of Adebayo et al., implement a ROAD baseline, and scale the explainability evaluation across more images and seeds.

---

## References

Abnar & Zuidema (2020), *Quantifying Attention Flow*; Sundararajan et al. (2017), *Integrated Gradients*; Selvaraju et al. (2017), *Grad-CAM*; Petsiuk et al. (2018), *RISE*; Samek et al. (2017), *AOPC*; Hooker et al. (2019), *ROAR*; Rong et al. (2022), *ROAD*; Chowdhury et al. (2020) & Rahman et al. (2021), *COVID-19 Radiography Database*.

---

## Author

**Parth Krishan Goswami** — AIMS-DTU Research Intern 2026 submission.
Trained weights & figures: `https://drive.google.com/file/d/1nzZghdzlxduhvil0xMVqCumB6bzbtUcf/view?`

<sub>Released for evaluation purposes. Dataset (c) its original authors.</sub>
