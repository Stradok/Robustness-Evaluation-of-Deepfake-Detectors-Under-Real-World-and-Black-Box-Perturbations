#!/usr/bin/env python3
"""
jpeg_attack.py

Runs JPEG compression attack on all trained models:
 - Xception
 - ResNet
 - MesoNet

Compression levels: [95, 85, 75, 60, 40, 20]

Outputs:
 - jpeg_attack_results.csv
 - examples/<attack_name_quality>/*.png

Run:
    python jpeg_attack.py
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io

import torch
import numpy as np
from torchvision import transforms

from config import Config
from dataset import create_dataloaders
from utils import get_device
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from models.xception import XceptionFakeDetect
from models.resnet import ResNetFakeDetect
from models.meso import MesoModel


# -----------------------
# Image Conversion Helpers
# -----------------------
def tensor_to_pil(t):
    t = t.detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(t)


def pil_to_tensor(img, device):
    return transforms.ToTensor()(img).to(device)


# -----------------------
# JPEG Compression Attack
# -----------------------
def jpeg_compress_tensor(x, quality, device):
    batch_output = []

    for img_tensor in x:
        pil = tensor_to_pil(img_tensor)

        # Compress into JPEG buffer
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Reload the compressed JPEG
        compressed = Image.open(buffer).convert("RGB")
        batch_output.append(pil_to_tensor(compressed, device))

    return torch.stack(batch_output, dim=0)


# -----------------------
# Load model
# -----------------------
def load_trained_model(name, device):
    name = name.lower()

    if name == "xception":
        model = XceptionFakeDetect(num_classes=2)
        ckpt = os.path.join(Config.SAVE_DIR, "xception", "best_xception.pth")

    elif name == "resnet":
        model = ResNetFakeDetect(num_classes=2)
        ckpt = os.path.join(Config.SAVE_DIR, "resnet", "best_resnet.pth")

    elif name == "meso":
        model = MesoModel(num_classes=2)
        ckpt = os.path.join(Config.SAVE_DIR, "meso", "best_meso.pth")

    else:
        raise ValueError("Invalid model")

    checkpoint = torch.load(ckpt, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# -----------------------
# Evaluation Metrics
# -----------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1


# -----------------------
# Main Function
# -----------------------
def main():

    device = get_device()
    print(f"[DEVICE] Using: {device}")

    # Load test set ONLY
    _, _, test_loader = create_dataloaders()
    print(f"[DATA] Test batches: {len(test_loader)}")

    # Models
    targets = ["xception", "resnet", "meso"]
    models = {name: load_trained_model(name, device) for name in targets}

    # JPEG quality levels
    qualities = [95, 85, 75, 60, 40, 20]

    # Output directories
    out_dir = Path("attacks_out") / "jpeg"
    examples_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "jpeg_attack_results.csv"


    rows = []
    fields = ["quality", "model", "accuracy", "precision", "recall", "f1"]

    print("\n[START] Running JPEG Compression Attacks...\n")

    for Q in qualities:
        attack_name = f"jpeg_q{Q}"
        print(f"[ATTACK] JPEG Quality = {Q}")

        preds_all = {m: [] for m in targets}
        labels_all = []

        saved_examples = 0
        example_dir = examples_dir / attack_name
        example_dir.mkdir(parents=True, exist_ok=True)

        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            attacked = jpeg_compress_tensor(imgs, Q, device)

            # Save a few attack examples
            if saved_examples < 10:
                for i in range(min(imgs.size(0), 10 - saved_examples)):
                    orig = tensor_to_pil(imgs[i])
                    adv = tensor_to_pil(attacked[i])
                    lab = labels[i].item()

                    orig.save(example_dir / f"{saved_examples}_orig_label{lab}.png")
                    adv.save(example_dir / f"{saved_examples}_jpeg_label{lab}.png")

                    saved_examples += 1

            # Evaluate each model
            for model_name in targets:
                model = models[model_name]

                with torch.no_grad():
                    out = model(attacked)
                    pred = torch.argmax(out, dim=1)

                preds_all[model_name].extend(pred.cpu().numpy())

            labels_all.extend(labels.cpu().numpy())

        # Compute metrics per model
        for model_name in targets:
            acc, prec, rec, f1 = compute_metrics(labels_all, preds_all[model_name])

            rows.append([Q, model_name, acc, prec, rec, f1])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)

    print("\n[DONE] JPEG Attack Results Saved:")
    print(f"CSV: {csv_path}")
    print(f"Examples: {examples_dir}\n")


if __name__ == "__main__":
    main()
