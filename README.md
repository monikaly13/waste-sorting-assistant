# Waste Sorting Assistant

A fine-tuned Vision Transformer (ViT) that classifies waste items into recyclable categories, built as a midterm project for [course name].

## Objective

Recycling infrastructure fails most often not because materials can't be recycled, but because they're never sorted correctly in the first place. Once waste is mixed, separating it again is expensive and rarely happens.

This project explores whether a lightweight, fine-tuned Vision Transformer can act as an automated first step in that sorting process — looking at a photo of a single waste item and returning both a material classification and a clear disposal instruction (recyclable vs. non-recyclable).

The goal isn't just to classify images accurately, but to frame the model as a **usable assistant**: something that could plausibly sit inside a smart bin, a sorting-facility camera, or a public-facing app to help close the sorting gap — a gap that is explicitly identified as one of the biggest structural failures in Cambodia's waste management system, where the vast majority of waste is landfilled or disposed of uncontrolled, and only a small fraction is recycled.

## Approach

- **Model:** `google/vit-base-patch16-224`, a Vision Transformer pretrained on ImageNet
- **Method:** Fine-tuning, not training from scratch — the pretrained model's general visual features are preserved, and only the final classification layer is retrained on waste categories
- **Dataset:** [TrashNet](https://github.com/garythung/trashnet) — 2,527 images across 6 categories (cardboard, glass, metal, paper, plastic, trash)
- **Why fine-tuning:** with limited compute (free-tier Colab GPU) and a relatively small dataset, fine-tuning a pretrained model is far more practical and reliable than training a transformer from scratch, while still achieving strong accuracy

## Pipeline

1. Load and inspect the dataset (class distribution, sample images)
2. Stratified 70/15/15 train/validation/test split
3. Load pretrained ViT and replace its classification head for 6 output classes
4. Preprocess images (resize to 224×224, normalize to match ViT's expected input)
5. Fine-tune using Hugging Face's `Trainer` API (8 epochs, learning rate 3e-5)
6. Evaluate on a held-out test set and analyze errors via a confusion matrix
7. Map predictions to a simple recyclable / non-recyclable instruction
8. Test on real, uncontrolled photos (not just dataset images) to check generalization

## Results

| Metric | Score |
|---|---|
| Validation accuracy | 96.3% |
| Test accuracy | 94.5% |

The confusion matrix shows the model's errors are concentrated between materials that share visual properties — glass, metal, and plastic are occasionally confused with each other (transparency, reflectivity), while cardboard and paper are sometimes confused due to shared texture and color. The smallest class, `trash`, has the fewest training examples and correspondingly the highest relative error rate.

Testing on original, non-dataset photos showed the model generalizes well to single, clearly-framed items (94%+ confidence on a real plastic bottle), but — as expected for a model trained on single-item images — it struggles with cluttered, multi-item scenes, since no single label can correctly describe a whole basket of mixed materials.

## Limitations

- TrashNet images are staged on clean backgrounds, not messy real-world scenes — a real deployment would need training data closer to actual bin/street conditions
- The model performs single-item classification only; multi-object scenes are out of scope
- "Recyclable" is treated as a simple per-category label here, though in practice recyclability also depends on material grade and local processing capability
- Trained on a relatively small, imbalanced dataset (some categories have far fewer images than others)

## Demo

The notebook includes a `predict_waste(image_path)` function that takes any image, returns the predicted category with a confidence score, and maps it to a disposal instruction (e.g. "Recyclable — Blue Bin" or "Non-recyclable — General Waste").

## Tech Stack

- Python, PyTorch
- Hugging Face `transformers`, `datasets`, `evaluate`
- Google Colab (free GPU tier)
