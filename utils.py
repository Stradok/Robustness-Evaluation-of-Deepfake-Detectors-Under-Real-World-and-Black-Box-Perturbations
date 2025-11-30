"""
utils.py
Utility helpers for training, evaluation, checkpointing, and reproducibility.

Designed to plug into:
 - config.Config (expects Config.IMG_SIZE, BATCH_SIZE, SAVE_DIR, etc.)
 - dataset.create_dataloaders() which returns train/val/test DataLoaders
 - model objects following PyTorch nn.Module API

Usage examples:
    from config import Config
    from utils import set_seed, get_device, save_checkpoint, load_checkpoint, accuracy
"""

import os
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import json

# -------------------------
# Reproducibility helpers
# -------------------------
def set_seed(seed: int = 42):
    """Set seeds for python, numpy and torch to ensure reproducible runs (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic flags (may slow training; optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Device / environment
# -------------------------
def get_device():
    """Return torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Checkpointing
# -------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict = None):
    """
    Save model + optimizer state.
    - model: nn.Module
    - optimizer: torch optimizer
    - epoch: current epoch number (int)
    - path: full filepath to save e.g. /content/InfoSec-Work/models/xception.pth
    - extra: optional dict with additional info (metrics, config)
    """
    ensure_dir(os.path.dirname(path))
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {}
    }
    torch.save(payload, path)

def load_checkpoint(path: str, model=None, optimizer=None, map_location=None):
    """
    Load checkpoint into model and optionally into optimizer.
    Returns: dict with keys: epoch, extra
    If model is provided, the state_dict will be loaded into it.
    """
    if map_location is None:
        map_location = get_device()
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {"epoch": ckpt.get("epoch", None), "extra": ckpt.get("extra", {})}


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy(outputs, targets):
    """
    Compute accuracy for binary/multi-class outputs.
    - outputs: logits or probabilities tensor shape (N, C)
    - targets: long tensor shape (N,)
    Returns: (accuracy_float, preds_tensor)
    """
    if outputs.dim() == 1 or outputs.shape[1] == 1:
        # binary single-logit case
        probs = torch.sigmoid(outputs.view(-1))
        preds = (probs > 0.5).long()
    else:
        preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0, preds


# -------------------------
# Simple training/validation step wrappers
# -------------------------
def train_one_epoch(model, dataloader, optimizer, device, criterion, scheduler=None, log_every=50):
    """
    Run one epoch of training.
    Returns: tuple (avg_loss, avg_acc)
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            # scheduler.step() usually called per epoch; for per-step schedulers call here
            pass

        running_loss += loss.item() * images.size(0)
        acc, _ = accuracy(outputs, labels)
        running_correct += acc * images.size(0)
        running_total += images.size(0)

        if log_every and (step + 1) % log_every == 0:
            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total
            print(f"[train] step {step+1}/{len(dataloader)} loss={avg_loss:.4f} acc={avg_acc:.4f}")

    avg_loss = running_loss / running_total if running_total > 0 else 0.0
    avg_acc = running_correct / running_total if running_total > 0 else 0.0
    return avg_loss, avg_acc


def validate(model, dataloader, device, criterion):
    """
    Run validation loop. Returns (avg_loss, avg_acc).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            acc, _ = accuracy(outputs, labels)
            total_correct += acc * images.size(0)
            total_n += images.size(0)
    avg_loss = total_loss / total_n if total_n > 0 else 0.0
    avg_acc = total_correct / total_n if total_n > 0 else 0.0
    return avg_loss, avg_acc


# -------------------------
# Logging helpers
# -------------------------
def make_run_id(prefix="run"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"

def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# -------------------------
# Quick utility to verify pipeline compatibility
# -------------------------
def quick_model_check(model, device=None):
    """
    Run a forward pass with random data matching Config.IMG_SIZE to ensure the model accepts the input size.
    Returns output shape.
    """
    from config import Config
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    bs = 2
    x = torch.randn(bs, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        out = model(x)
    return out.shape
