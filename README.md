# ♻️ Waste Sorting Assistant

A fine-tuned Vision Transformer (ViT) that looks at a photo of a waste item, identifies the material, and tells you how to dispose of it.

**Test accuracy: 98.6%** · 10 waste categories · Runs on a free Google Colab GPU

---

## Overview

Recycling often fails at the first step: waste is never sorted correctly, and once it is mixed it is expensive to separate again. This is a major gap in Cambodia's waste management, where most waste is landfilled and only a small fraction is recycled.

This project tests whether a lightweight image model can act as an automated first sorting step, for example inside a smart bin, a sorting-facility camera, or a public app.

**Input:** a photo of a single item
**Output:** a material label, a confidence score, and a disposal instruction

> Example: `Plastic` + a confidence score + a disposal instruction

---

## How It Works

| | |
|---|---|
| **Model** | `google/vit-base-patch16-224` (pretrained on ImageNet) |
| **Method** | Fine-tuning: only the final classification layer is retrained for waste categories |
| **Dataset** | [Garbage Classification (Kaggle)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification): about 10,900 images (three glass colors merged into one `glass` class) |
| **Classes (10)** | battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash |
| **Split** | 70% train / 15% validation / 15% test (stratified) |
| **Training** | Hugging Face `Trainer`, 8 epochs, learning rate 3e-5, batch size 16, best checkpoint kept (about 44 min on a Colab GPU) |

Fine-tuning a pretrained model is far more practical than training from scratch when compute and data are limited.

---

## Results

| Metric | Score |
|---|---|
| Validation accuracy | 98.0% (best epoch) |
| Test accuracy | 98.6% |

- A confusion matrix in the notebook shows which materials get mixed up.
- Built for single, clearly framed items. Cluttered scenes with several items are not supported.

---

## Getting Started

### Prerequisite: Kaggle API key

The dataset is downloaded from Kaggle, so you need a free account and an API key:

1. Sign in at [kaggle.com](https://www.kaggle.com/) → **Settings** → **API** → create a token. This downloads `kaggle.json`.
2. Keep this file private. Never commit it to GitHub.

### Option A: Google Colab (easiest, no setup)

1. Open `train_vit.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Open notebook → GitHub tab, then paste this repo URL).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells: **Runtime → Run all**. When prompted, upload your `kaggle.json`.

### Option B: Run locally

**Requirements:** Python 3.9+, Git. A GPU is recommended; CPU works but training is slow.

```bash
# 1. Clone the repo
git clone https://github.com/monikaly13/waste-sorting-assistant.git
cd waste-sorting-assistant

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch transformers datasets evaluate scikit-learn accelerate pillow matplotlib kaggle jupyter

# 4. Add your Kaggle key
mkdir -p ~/.kaggle && mv /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 5. Launch the notebook
jupyter notebook train_vit.ipynb
```

The notebook uses Colab-specific code (`google.colab`, `/content/...` paths) in its download cell. If you run locally, change that cell to download and unzip into a local folder instead.

Then run the cells in order, from top to bottom.

---

## Try a Prediction

After training, the notebook provides a helper function:

```python
predict_waste("path/to/your_image.jpg")
```

It returns the predicted category, a confidence score, and a disposal instruction.

---

## Project Structure

```
waste-sorting-assistant/
├── train_vit.ipynb    # Main notebook: data, training, evaluation, demo
├── notebooks/         # Supporting notebooks
└── README.md
```

---

## Limitations

- Single-item classification only. Multi-item scenes are out of scope.
- "Recyclable" is a simple per-category label. In reality it also depends on material grade and local processing capability.
- Dataset images may not match real bin or street conditions, and some classes have far fewer images than others.

---

## Tech Stack

Python · PyTorch · Hugging Face (`transformers`, `datasets`, `evaluate`) · Kaggle API · Google Colab

---

## Acknowledgements

- Dataset: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Abla (Kaggle)
- Base model: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)

---

## Author

**Monika Ly** · [GitHub](https://github.com/monikaly13)
