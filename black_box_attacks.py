#!/usr/bin/env python3
"""
full_attack_pipeline.py

Unified pipeline:
 - Black-box attacks: gaussian noise, salt&pepper, gaussian blur, brightness/contrast
 - Transfer PGD (crafted on surrogate, tested on targets)
 - JPEG compression (qualities: 85, 60, 40, 20)

Outputs:
 - attacks_out/full_attack_results.csv  (tall format)
 - attacks_out/examples/<attack>_<param>/{000_orig_labelX.png,000_adv_labelX.png}
 - prints progress/logs

Usage:
    python full_attack_pipeline.py --surrogate resnet --device cuda --out_dir attacks_out --overwrite
"""

import os
import argparse
from pathlib import Path
import csv
import random
import shutil
from tqdm import tqdm

import math
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# project imports - assumes these modules exist in your repo
from config import Config
from dataset import create_dataloaders
from utils import get_device

from models.xception import XceptionFakeDetect
from models.resnet import ResNetFakeDetect
from models.meso import MesoModel

# ---------------------------
# Image helpers
# ---------------------------
_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()

def tensor_to_pil(tensor):
    """tensor: [C,H,W] in 0..1"""
    return _to_pil(tensor.detach().cpu().clamp(0,1))

def pil_to_tensor(pil_img, device):
    return _to_tensor(pil_img).to(device)

def pil_to_uint8_array(pil_img):
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)

# ---------------------------
# Attacks (tensor in 0..1)
# ---------------------------
def gaussian_noise_tensor(x, sigma):
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0,1)

def salt_pepper_tensor(x, amount):
    x = x.clone()
    B, C, H, W = x.shape
    n_pixels = int(amount * H * W)
    for b in range(B):
        coords = torch.randperm(H * W)[:n_pixels]
        ys = coords // W
        xs = coords % W
        for y, xx in zip(ys.tolist(), xs.tolist()):
            if random.random() < 0.5:
                x[b, :, y, xx] = 0.0
            else:
                x[b, :, y, xx] = 1.0
    return x

def brightness_contrast_tensor(x, brightness=1.0, contrast=1.0):
    device = x.device
    out = []
    for img in x:
        pil = tensor_to_pil(img)
        if brightness != 1.0:
            pil = ImageEnhance.Brightness(pil).enhance(brightness)
        if contrast != 1.0:
            pil = ImageEnhance.Contrast(pil).enhance(contrast)
        out.append(pil_to_tensor(pil, device))
    return torch.stack(out, dim=0)

def gaussian_blur_tensor(x, radius=1.5):
    device = x.device
    out = []
    for img in x:
        pil = tensor_to_pil(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        out.append(pil_to_tensor(pil, device))
    return torch.stack(out, dim=0)

def jpeg_compress_tensor(x, quality, device):
    out = []
    for img in x:
        pil = tensor_to_pil(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")
        out.append(pil_to_tensor(comp, device))
    return torch.stack(out, dim=0)

# ---------------------------
# PGD on surrogate (white-box to surrogate)
# ---------------------------
def pgd_on_surrogate(surrogate, images, labels, eps=8/255, alpha=2/255, steps=10, device='cpu', targeted=False):
    surrogate.eval()
    images = images.clone().to(device)
    labels = labels.clone().to(device)
    adv = images.clone().detach()
    adv.requires_grad = True
    orig = images.clone().detach()
    for _ in range(steps):
        outputs = surrogate(adv)
        loss = F.cross_entropy(outputs, labels)
        if targeted:
            loss = -loss
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() + alpha * torch.sign(grad.detach())
        adv = torch.max(torch.min(adv, orig + eps), orig - eps)
        adv = adv.clamp(0,1)
        adv.requires_grad = True
    return adv.detach()

# ---------------------------
# Model loading
# ---------------------------
def load_trained_model(name, device):
    name = name.lower()
    if name == "xception":
        m = XceptionFakeDetect(num_classes=2)
        ckpt = Path(Config.SAVE_DIR) / "xception" / "best_xception.pth"
    elif name == "resnet":
        m = ResNetFakeDetect(num_classes=2)
        ckpt = Path(Config.SAVE_DIR) / "resnet" / "best_resnet.pth"
    elif name == "meso":
        m = MesoModel(num_classes=2)
        ckpt = Path(Config.SAVE_DIR) / "meso" / "best_meso.pth"
    else:
        raise ValueError("unknown model")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        m.load_state_dict(state["model_state_dict"])
    else:
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

# ---------------------------
# Metrics
# ---------------------------
def compute_class_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return (float("nan"),)*4
    acc = float((y_true == y_pred).mean())
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return acc, prec, rec, f1

def compute_psnr_ssim_pair(orig_pil, adv_pil):
    orig = pil_to_uint8_array(orig_pil)
    adv = pil_to_uint8_array(adv_pil)
    # ensure shape match
    if orig.shape != adv.shape:
        # resize adv to orig
        adv = np.array(adv_pil.convert(orig_pil.mode).resize(orig_pil.size), dtype=np.uint8)
    psnr = peak_signal_noise_ratio(orig, adv, data_range=255)
    ssim = structural_similarity(orig, adv, channel_axis=2)
    return float(psnr), float(ssim)

# ---------------------------
# Main pipeline
# ---------------------------
def run(args):
    device = args.device if args.device else get_device()
    print(f"[INFO] Device: {device}")

    # create loaders
    _, _, test_loader = create_dataloaders()
    print(f"[INFO] Test loader batches: {len(test_loader)}")

    targets = ["xception", "resnet", "meso"]
    print("[INFO] Loading target models")
    target_models = {t: load_trained_model(t, device) for t in targets}

    print(f"[INFO] Loading surrogate model: {args.surrogate}")
    surrogate = load_trained_model(args.surrogate, device)
    surrogate.eval()

    out_base = Path(args.out_dir)
    if out_base.exists() and args.overwrite:
        shutil.rmtree(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    examples_root = out_base / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / "full_attack_results.csv"
    csv_fields = [
        "attack", "attack_param", "target",
        "accuracy", "precision", "recall", "f1",
        "mean_psnr", "mean_ssim", "num_examples"
    ]
    csv_rows = []

    # attack parameter lists
    gaussian_sigmas = args.gauss_sigmas
    sp_amounts = args.sp_amounts
    blur_radii = args.blur_radii
    bright_factors = args.brightness_factors
    transfer_eps = args.transfer_eps
    transfer_alpha = args.transfer_alpha
    transfer_steps = args.transfer_steps
    jpeg_qualities = args.jpeg_qualities

    # helper to run an attack function that returns attacked tensor
    global_example_id = 0
    def process_attack_batch(attack_name, param_str, attacked_batch, orig_batch, labels, save_examples_limit=8):
        nonlocal global_example_id
        # attacked_batch, orig_batch: tensors on device (B,C,H,W) 0..1
        B = orig_batch.size(0)
        # evaluate each model and collect preds
        per_model_preds = {t: [] for t in targets}
        y_labels = []
        # PSNR/SSIM aggregated per model (we compute per-saved-example only)
        psnr_list = {t: [] for t in targets}
        ssim_list = {t: [] for t in targets}

        # For saving examples we save up to save_examples_limit overall per attack (not per-model)
        saved_examples = 0

        # iterate batch
        for b in range(B):
            orig_t = orig_batch[b].detach().cpu()
            adv_t = attacked_batch[b].detach().cpu()
            label = int(labels[b].item())
            y_labels.append(label)

            # evaluate models
            for t in targets:
                model = target_models[t]
                with torch.no_grad():
                    out = model(adv_t.unsqueeze(0).to(device))
                    pred = int(out.argmax(dim=1).cpu().item())
                per_model_preds[t].append(pred)

            # save examples and compute psnr/ssim on the saved ones
            if saved_examples < save_examples_limit:
                # produce PILs
                orig_pil = tensor_to_pil(orig_t)
                adv_pil = tensor_to_pil(adv_t)
                # save under examples folder
                save_folder = examples_root / f"{attack_name}_{param_str}"
                save_folder.mkdir(parents=True, exist_ok=True)

                fname_base = f"{global_example_id:06d}"
                orig_path = save_folder / f"{fname_base}_orig_label{label}.png"
                adv_path = save_folder / f"{fname_base}_adv_label{label}.png"
                orig_pil.save(orig_path)
                adv_pil.save(adv_path)

                # compute PSNR/SSIM
                psnr_val, ssim_val = compute_psnr_ssim_pair(orig_pil, adv_pil)
                # append same psnr/ssim to all models' lists (we average per-model later over saved examples)
                for t in targets:
                    psnr_list[t].append(psnr_val)
                    ssim_list[t].append(ssim_val)

                saved_examples += 1
                global_example_id += 1

        # convert preds/labels to numpy arrays and compute metrics per model
        for t in targets:
            preds = np.array(per_model_preds[t], dtype=np.int32)
            labels_arr = np.array(y_labels, dtype=np.int32)
            acc, prec, rec, f1 = compute_class_metrics(labels_arr, preds)
            # average psnr/ssim for this model (if no saved examples, NaN)
            mean_psnr = float(np.mean(psnr_list[t])) if len(psnr_list[t])>0 else float("nan")
            mean_ssim = float(np.mean(ssim_list[t])) if len(ssim_list[t])>0 else float("nan")
            csv_rows.append([
                attack_name, param_str, t,
                acc, prec, rec, f1,
                mean_psnr, mean_ssim, len(psnr_list[t])
            ])

    # ---------------------------
    # Run each attack sweep
    # ---------------------------
    print("[INFO] Running Gaussian noise attacks")
    for sigma in gaussian_sigmas:
        param_str = f"sigma{sigma}"
        for imgs, labels in tqdm(test_loader, desc=f"gauss_{sigma}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            attacked = gaussian_noise_tensor(imgs, sigma)
            process_attack_batch("gaussian_noise", param_str, attacked, imgs, labels)

    print("[INFO] Running Salt&Pepper attacks")
    for amt in sp_amounts:
        param_str = f"amt{amt}"
        for imgs, labels in tqdm(test_loader, desc=f"sp_{amt}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            attacked = salt_pepper_tensor(imgs, amount=amt)
            process_attack_batch("salt_pepper", param_str, attacked, imgs, labels)

    print("[INFO] Running Gaussian Blur attacks")
    for r in blur_radii:
        param_str = f"r{r}"
        for imgs, labels in tqdm(test_loader, desc=f"blur_{r}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            attacked = gaussian_blur_tensor(imgs, radius=r)
            process_attack_batch("gaussian_blur", param_str, attacked, imgs, labels)

    print("[INFO] Running Brightness/Contrast attacks")
    for bf in bright_factors:
        cf = 1.0
        param_str = f"b{bf}_c{cf}"
        for imgs, labels in tqdm(test_loader, desc=f"bright_{bf}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            attacked = brightness_contrast_tensor(imgs, brightness=bf, contrast=cf)
            process_attack_batch("brightness_contrast", param_str, attacked, imgs, labels)

    print("[INFO] Running Transfer PGD attacks (surrogate -> targets)")
    for eps in transfer_eps:
        param_str = f"eps{eps}"
        for imgs, labels in tqdm(test_loader, desc=f"pgd_eps{eps}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            adv = pgd_on_surrogate(surrogate, imgs, labels, eps=eps, alpha=transfer_alpha, steps=transfer_steps, device=device)
            process_attack_batch("transfer_pgd", param_str, adv, imgs, labels)

    print("[INFO] Running JPEG compression attacks")
    for q in jpeg_qualities:
        param_str = f"q{q}"
        for imgs, labels in tqdm(test_loader, desc=f"jpeg_q{q}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            attacked = jpeg_compress_tensor(imgs, q, device)
            process_attack_batch("jpeg", param_str, attacked, imgs, labels)

    # ---------------------------
    # Write CSV
    # ---------------------------
    print(f"[INFO] Writing CSV to {csv_path}")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(csv_fields)
        for row in csv_rows:
            writer.writerow(row)

    print("[DONE] Pipeline finished.")
    print(f"CSV: {csv_path}")
    print(f"Examples root: {examples_root}")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate", type=str, default="resnet", choices=["xception","resnet","meso"])
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default auto)")
    parser.add_argument("--out_dir", type=str, default="attacks_out", help="output directory")
    parser.add_argument("--overwrite", action="store_true", help="overwrite out_dir if exists")
    parser.add_argument("--gauss_sigmas", nargs="+", type=float, default=[0.005, 0.01, 0.02], help="gaussian noise sigmas")
    parser.add_argument("--sp_amounts", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--blur_radii", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--brightness_factors", nargs="+", type=float, default=[0.8, 0.9, 1.1])
    parser.add_argument("--transfer_eps", nargs="+", type=float, default=[4/255, 8/255])
    parser.add_argument("--transfer_alpha", type=float, default=2/255)
    parser.add_argument("--transfer_steps", type=int, default=10)
    parser.add_argument("--jpeg_qualities", nargs="+", type=int, default=[85, 60, 40, 20])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # convert list args typed as strings (if necessary)
    args.gauss_sigmas = list(map(float, args.gauss_sigmas)) if isinstance(args.gauss_sigmas, list) else args.gauss_sigmas
    args.sp_amounts = list(map(float, args.sp_amounts)) if isinstance(args.sp_amounts, list) else args.sp_amounts
    args.blur_radii = list(map(float, args.blur_radii)) if isinstance(args.blur_radii, list) else args.blur_radii
    args.brightness_factors = list(map(float, args.brightness_factors)) if isinstance(args.brightness_factors, list) else args.brightness_factors
    args.transfer_eps = list(map(float, args.transfer_eps)) if isinstance(args.transfer_eps, list) else args.transfer_eps
    args.jpeg_qualities = list(map(int, args.jpeg_qualities)) if isinstance(args.jpeg_qualities, list) else args.jpeg_qualities

    run(args)
