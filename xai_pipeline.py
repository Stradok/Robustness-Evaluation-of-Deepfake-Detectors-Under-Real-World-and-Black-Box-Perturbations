#!/usr/bin/env python3
"""
xai_pipeline.py

Generate XAI visualizations for:
 - Clean images
 - Adversarial images

Methods:
 - Grad-CAM
 - ScoreCAM
 - Guided Backprop
 - Integrated Gradients

Models supported:
 - Xception
 - ResNet
 - MesoNet

Automatically finds the last Conv2D layer for GradCAM & ScoreCAM.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from torchvision import transforms

# XAI imports (latest API)
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from captum.attr import IntegratedGradients, GuidedBackprop

# Your project imports
from config import Config
from utils import get_device
from dataset import get_transforms

from models.xception import XceptionFakeDetect
from models.resnet import ResNetFakeDetect
from models.meso import MesoModel


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def denormalize_tensor(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    t = t.cpu() if t.is_cuda else t
    return t * std + mean

def tensor_to_img(t):
    """Converts a normalized tensor to a numpy image in [0,1]."""
    # Denormalize first (dataset uses mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    t_denorm = denormalize_tensor(t)
    t_np = t_denorm.detach().cpu().permute(1, 2, 0).numpy()
    t_np = np.clip(t_np, 0, 1)
    return t_np

def load_image_pair(orig_path, adv_path, device):
    """Load original and adversarial image pair with proper transforms."""
    transform = get_transforms()
    
    orig_img = Image.open(orig_path).convert("RGB")
    adv_img = Image.open(adv_path).convert("RGB")
    
    orig_tensor = transform(orig_img).unsqueeze(0).to(device)
    adv_tensor = transform(adv_img).unsqueeze(0).to(device)
    
    # Get display versions (denormalized)
    orig_display = tensor_to_img(orig_tensor[0])
    adv_display = tensor_to_img(adv_tensor[0])
    
    return orig_tensor, adv_tensor, orig_display, adv_display

def find_last_conv_layer(model):
    """Automatically find the last Conv2D layer for CAM."""
    # Try using model's method if available (Xception, MesoNet)
    if hasattr(model, 'get_last_conv_module'):
        layer = model.get_last_conv_module()
        if layer is not None:
            return layer
    
    # Fallback: find last Conv2d (for ResNet)
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

def find_attack_pairs(limit_per_attack=10):
    """Find (orig_path, adv_path) pairs from attacks_out directories."""
    pairs = []
    base = Path("attacks_out")
    
    # Check attacks_out/examples/
    examples_dir = base / "examples"
    if examples_dir.exists():
        for attack_folder in sorted(examples_dir.iterdir()):
            if not attack_folder.is_dir():
                continue
            origs = sorted([p for p in attack_folder.iterdir() 
                          if "orig" in p.name and p.suffix == ".png"])
            advs = sorted([p for p in attack_folder.iterdir() 
                         if "adv" in p.name and p.suffix == ".png"])
            
            for orig in origs[:limit_per_attack]:
                # Find matching adv file by prefix
                prefix = orig.name.split("_")[0]
                adv = None
                for a in advs:
                    if a.name.startswith(prefix):
                        adv = a
                        break
                if adv:
                    pairs.append((orig, adv, attack_folder.name))
    
    # Check attacks_out/jpeg/examples/
    jpeg_dir = base / "jpeg" / "examples"
    if jpeg_dir.exists():
        for qfolder in sorted(jpeg_dir.iterdir()):
            if not qfolder.is_dir():
                continue
            origs = sorted([p for p in qfolder.iterdir() 
                          if "orig" in p.name and p.suffix == ".png"])
            advs = sorted([p for p in qfolder.iterdir() 
                         if ("jpeg" in p.name or "adv" in p.name) and p.suffix == ".png"])
            
            for orig in origs[:limit_per_attack]:
                prefix = orig.name.split("_")[0]
                adv = None
                for a in advs:
                    if a.name.startswith(prefix):
                        adv = a
                        break
                if adv:
                    pairs.append((orig, adv, qfolder.name))
    
    return pairs


# ---------------------------------------------------------
# Load trained model
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
        raise ValueError(f"Unknown model: {name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Main XAI function
# ---------------------------------------------------------
def generate_xai_for_model(model, model_name, pairs, device, output_dir):
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        print(f"[ERROR] Could not find Conv2d layer for {model_name}")
        return
    
    print(f"[XAI] {model_name}: using target layer → {target_layer}")

    # Initialize CAM methods (remove use_cuda parameter - auto-detects)
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    scorecam = ScoreCAM(model=model, target_layers=[target_layer])

    # Initialize Captum methods
    gbp = GuidedBackprop(model)
    ig = IntegratedGradients(model)

    for idx, (orig_path, adv_path, attack_name) in enumerate(tqdm(pairs, desc=f"{model_name} XAI")):
        try:
            # Load image pair
            clean_tensor, adv_tensor, clean_img, adv_img = load_image_pair(
                orig_path, adv_path, device
            )
            
            # Get prediction for target class
            with torch.no_grad():
                clean_out = model(clean_tensor)
                target_class = int(torch.argmax(clean_out, dim=1).item())
            
            targets = [ClassifierOutputTarget(target_class)]
            
            # Create subdirectory for this attack type
            attack_dir = output_dir / attack_name
            attack_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = f"{idx:03d}"

            # -------------------------
            # GRAD-CAM
            # -------------------------
            try:
                cam_clean = gradcam(input_tensor=clean_tensor, targets=targets)[0]
                cam_adv = gradcam(input_tensor=adv_tensor, targets=targets)[0]

                cam_clean_img = show_cam_on_image(clean_img, cam_clean, use_rgb=True)
                cam_adv_img = show_cam_on_image(adv_img, cam_adv, use_rgb=True)

                Image.fromarray(cam_clean_img).save(
                    attack_dir / f"{base_name}_gradcam_clean.png"
                )
                Image.fromarray(cam_adv_img).save(
                    attack_dir / f"{base_name}_gradcam_adv.png"
                )

            except Exception as e:
                print(f"  GradCAM error: {e}")

            # -------------------------
            # SCORECAM
            # -------------------------
            try:
                sc_clean = scorecam(input_tensor=clean_tensor, targets=targets)[0]
                sc_adv = scorecam(input_tensor=adv_tensor, targets=targets)[0]

                sc_clean_img = show_cam_on_image(clean_img, sc_clean, use_rgb=True)
                sc_adv_img = show_cam_on_image(adv_img, sc_adv, use_rgb=True)

                Image.fromarray(sc_clean_img).save(
                    attack_dir / f"{base_name}_scorecam_clean.png"
                )
                Image.fromarray(sc_adv_img).save(
                    attack_dir / f"{base_name}_scorecam_adv.png"
                )

            except Exception as e:
                print(f"  ScoreCAM error: {e}")

            # -------------------------
            # GUIDED BACKPROP
            # -------------------------
            try:
                guided_clean = gbp.attribute(clean_tensor, target=target_class)
                guided_adv = gbp.attribute(adv_tensor, target=target_class)
                
                # Process clean
                guided_clean_np = guided_clean[0].detach().cpu().numpy().transpose(1,2,0)
                guided_clean_np = np.abs(guided_clean_np)
                guided_clean_np = (guided_clean_np - guided_clean_np.min()) / (guided_clean_np.max() - guided_clean_np.min() + 1e-8)
                Image.fromarray((guided_clean_np * 255).astype(np.uint8)).save(
                    attack_dir / f"{base_name}_guidedbp_clean.png"
                )
                
                # Process adv
                guided_adv_np = guided_adv[0].detach().cpu().numpy().transpose(1,2,0)
                guided_adv_np = np.abs(guided_adv_np)
                guided_adv_np = (guided_adv_np - guided_adv_np.min()) / (guided_adv_np.max() - guided_adv_np.min() + 1e-8)
                Image.fromarray((guided_adv_np * 255).astype(np.uint8)).save(
                    attack_dir / f"{base_name}_guidedbp_adv.png"
                )
            except Exception as e:
                print(f"  GuidedBP error: {e}")

            # -------------------------
            # INTEGRATED GRADIENTS
            # -------------------------
            try:
                ig_clean = ig.attribute(clean_tensor, target=target_class, n_steps=50)
                ig_adv = ig.attribute(adv_tensor, target=target_class, n_steps=50)
                
                # Process clean
                ig_clean_np = ig_clean[0].detach().cpu().numpy().transpose(1,2,0)
                ig_clean_np = np.abs(ig_clean_np)
                ig_clean_np = ig_clean_np / (ig_clean_np.max() + 1e-8)
                Image.fromarray((ig_clean_np * 255).astype(np.uint8)).save(
                    attack_dir / f"{base_name}_ig_clean.png"
                )
                
                # Process adv
                ig_adv_np = ig_adv[0].detach().cpu().numpy().transpose(1,2,0)
                ig_adv_np = np.abs(ig_adv_np)
                ig_adv_np = ig_adv_np / (ig_adv_np.max() + 1e-8)
                Image.fromarray((ig_adv_np * 255).astype(np.uint8)).save(
                    attack_dir / f"{base_name}_ig_adv.png"
                )
            except Exception as e:
                print(f"  Integrated Gradients error: {e}")
                
        except Exception as e:
            print(f"  Error processing pair {idx} ({attack_name}): {e}")
            import traceback
            traceback.print_exc()
            continue


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Find attack example pairs from attacks_out directories
    pairs = find_attack_pairs(limit_per_attack=10)
    if len(pairs) == 0:
        print("[ERROR] No attack example pairs found in attacks_out/examples or attacks_out/jpeg/examples")
        print("  Please run attack scripts first (black_box_attacks.py, jpeg_attack.py)")
        return
    
    print(f"[INFO] Found {len(pairs)} image pairs to process")

    models = ["xception", "resnet", "meso"]

    for m in models:
        try:
            print(f"\n[INFO] Loading model: {m}")
            model = load_model(m, device)
            generate_xai_for_model(model, m, pairs, device, "xai_outputs")
        except Exception as e:
            print(f"[ERROR] Failed to process {m}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n[INFO] XAI pipeline complete!")


if __name__ == "__main__":
    main()