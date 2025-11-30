"""
train.py
Training module designed for use INSIDE a Colab Python notebook.

Usage:
    from train import train
    model = train("xception")
"""

import os
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from utils import (
    set_seed, get_device, save_checkpoint,
    validate
)

# Model imports
from models.xception import XceptionFakeDetect
from models.resnet import ResNetFakeDetect
from models.meso import MesoModel


# ----------------------------------------------------
# Model Loader
# ----------------------------------------------------
def load_model(name: str, device):

    name = name.lower()

    if name == "xception":
        print(colored("[INFO] Loading Xception model...", "cyan"))
        model = XceptionFakeDetect(num_classes=2)

    elif name == "resnet":
        from models.resnet import ResNetFakeDetect
        print(colored("[INFO] Loading ResNet model...", "cyan"))
        model = ResNetFakeDetect(num_classes=2)

    elif name == "meso":
        from models.meso import MesoModel
        print(colored("[INFO] Loading MesoNet model...", "cyan"))
        model = MesoModel(num_classes=2)

    else:
        raise ValueError("Unknown model name. Choose: xception, resnet, meso")

    return model.to(device)


# ----------------------------------------------------
# MAIN TRAINING FUNCTION
# ----------------------------------------------------
def train(model_name: str):

    # 0. Setup
    set_seed(42)   # Config does not contain SEED, so use fixed seed
    device = get_device()
    print(colored(f"[DEVICE] Using: {device}", "green"))

    # 1. Load data
    train_loader, val_loader, test_loader = create_dataloaders()
    print(colored(
        f"[DATA] Train={len(train_loader)}  Val={len(val_loader)}  Test={len(test_loader)}",
        "yellow"
    ))

    # 2. Load model
    model = load_model(model_name, device)

    # 3. Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # 4. Model save folder
    model_folder = os.path.join(Config.SAVE_DIR, model_name)
    os.makedirs(model_folder, exist_ok=True)

    best_val_acc = 0.0

    print(colored("\n[START TRAINING]\n", "magenta"))

    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------
    for epoch in range(1, Config.NUM_EPOCHS + 1):

        print(colored(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}", "blue"))

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        with tqdm(total=len(train_loader), desc="Training", ncols=100) as bar:

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct = (preds == labels).sum().item()

                epoch_loss += loss.item() * images.size(0)
                epoch_correct += correct
                epoch_total += images.size(0)

                bar.update(1)

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total

        # ---- VALIDATION ----
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        print(colored(f"Train Loss: {train_loss:.4f}   Train Acc: {train_acc:.4f}", "green"))
        print(colored(f"Val Loss:   {val_loss:.4f}    Val Acc:   {val_acc:.4f}", "yellow"))

        # Save model if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(model_folder, f"best_{model_name}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(colored(f"[CHECKPOINT SAVED] {ckpt_path}", "cyan"))

    # ------------------------------------------------
    # TEST BEST MODEL
    # ------------------------------------------------
    print(colored("\n[TESTING BEST MODEL]\n", "magenta"))

    best_ckpt = os.path.join(model_folder, f"best_{model_name}.pth")
    state = torch.load(best_ckpt, map_location=device)

    model.load_state_dict(state["model_state_dict"])

    test_loss, test_acc = validate(model, test_loader, device, criterion)

    print(colored(f"\nTEST LOSS = {test_loss:.4f}", "red"))
    print(colored(f"TEST ACC  = {test_acc:.4f}", "red"))

    print(colored("\n[TRAINING COMPLETE]", "magenta"))

    return model


if __name__ == "__main__":
    import argparse

    print(">>> Starting train.py from terminal...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["xception", "resnet", "meso"],
                        help="Choose which model to train")

    args = parser.parse_args()

    print(f">>> Model selected: {args.model}")
    print(">>> Calling train()...\n")

    try:
        train(args.model)
        print("\n>>> Training completed successfully!")
    except Exception as e:
        print("\n[ERROR] Something went wrong:")
        print(str(e))
