# Horse vs. Zebra Detector — Tricky Image Classification

Advanced Machine Learning Project

## Overview

Three image classifiers trained to distinguish horses from zebras, then
stress-tested on an AI-generated "tricky" set built around
texture-vs-shape conflicts (e.g. a horse wearing zebra-striped blankets).

| Model | Type |
|---|---|
| SimpleCNN | From-scratch CNN (3 conv blocks + BN + dropout) |
| EfficientNet-B0 | CNN, transfer learning (ImageNet) |
| DeiT (`facebook/deit-base-patch16-224`) | Transformer, transfer learning (ImageNet) |

## Dataset

| | Horse | Zebra | Total |
|---|---|---|---|
| Images (after duplicate removal) | 856 | 849 | 1,705 |

Sources: Unsplash, Pexels, Pixabay, Wikimedia Commons (freely licensed).
7 exact duplicates were removed via MD5 hashing before splitting.

## Pipeline

1. Duplicate removal (MD5 hashing) before splitting — prevents data leakage
2. Train / val / test split (70/15/15, `random_state=42`)
3. Image resize (256×256) + save to Drive for fast reload across sessions
4. Cross-split leakage check
5. Shared `train_model` / `evaluate_model` functions for all three architectures
6. Evaluation on both the normal test set and a 100-image AI-generated tricky set

## Results

| Model | Normal Test Accuracy | Tricky (AI) Test Accuracy |
|---|---|---|
| SimpleCNN | 92.6% | 66.0% |
| EfficientNet-B0 | 99.6% | 81.0% |
| DeiT | 99.6% | 86.0% |

All three models perform well on the normal test set, but accuracy drops
noticeably on the AI-generated tricky set — least for DeiT, most for the
from-scratch CNN. This gap is used as a proxy for how much each model
relies on genuine shape understanding versus superficial texture cues.

## Repository Structure

```
horse-vs-zebra-app/
├── starter.ipynb              # full training + evaluation pipeline (run on Google Colab, GPU)
├── api_image_downloader.ipynb # dataset collection
├── app.py                     # Streamlit demo (3-model live comparison), built locally in VS Code
├── requirements.txt           # Python dependencies
└── models/                    # trained .pth weights, exported from Colab
```

`starter.ipynb` and `api_image_downloader.ipynb` were developed and
trained entirely on **Google Colab** (GPU runtime) — together they
produce the three `.pth` model files. `app.py` is a separate,
locally-built **Streamlit** app that loads those exported weights to run
live inference outside Colab.

## Run the Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

`Python` `PyTorch` `torchvision` `Hugging Face Transformers / timm` `EfficientNet` `DeiT (Vision Transformer)` `Streamlit` `PIL / Pillow` `NumPy` `Matplotlib`

**Topics (for GitHub About section):** `computer-vision` `pytorch` `transfer-learning` `vision-transformer` `image-classification` `robustness` `deep-learning`
