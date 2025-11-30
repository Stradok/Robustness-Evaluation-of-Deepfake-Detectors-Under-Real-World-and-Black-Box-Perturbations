"""
evaluate.py

Evaluates all trained models on the SAME dataset and
prints a comparison table:

- Xception
- ResNet
- MesoNet

Metrics:
    Accuracy
    Precision
    Recall
    F1 Score

Run:
    python evaluate.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from dataset import create_dataloaders
from utils import get_device

# Models
from models.xception import XceptionFakeDetect
from models.resnet import ResNetFakeDetect
from models.meso import MesoModel


# ---------------------------------------------------------
# LOAD BEST MODEL
# ---------------------------------------------------------
def load_model(name, device):

    if name == "xception":
        model = XceptionFakeDetect(num_classes=2)
        path = os.path.join(Config.SAVE_DIR, "xception", "best_xception.pth")

    elif name == "resnet":
        model = ResNetFakeDetect(num_classes=2)
        path = os.path.join(Config.SAVE_DIR, "resnet", "best_resnet.pth")

    elif name == "meso":
        model = MesoModel(num_classes=2)
        path = os.path.join(Config.SAVE_DIR, "meso", "best_meso.pth")

    else:
        raise ValueError("Invalid model name")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader, device):

    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall    = recall_score(all_labels, all_preds, average="binary")
    f1        = f1_score(all_labels, all_preds, average="binary")

    return accuracy, precision, recall, f1


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    device = get_device()
    print(f"\n[DEVICE] Using: {device}")

    # We will use VAL loader for evaluation
    _, val_loader, _ = create_dataloaders()
    print(f"[INFO] Using VAL dataset: {len(val_loader)} batches")

    models = ["xception", "resnet", "meso"]
    results = []

    for name in models:

        print(f"\n[INFO] Loading {name.upper()}...")
        model = load_model(name, device)

        acc, prec, rec, f1 = evaluate_model(model, val_loader, device)

        results.append([name, acc, prec, rec, f1])

    # ---------------------------------------------------------
    # PRINT COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'MODEL':<12}{'ACCURACY':<12}{'PRECISION':<12}{'RECALL':<12}{'F1-SCORE'}")
    print("="*60)

    for row in results:
        name, acc, prec, rec, f1 = row
        print(f"{name:<12}{acc*100:>9.2f}%   {prec:>8.4f}   {rec:>8.4f}   {f1:>8.4f}")

    print("="*60)
    print("\n✅ Evaluation complete!\n")

if __name__ == "__main__":
    main()
