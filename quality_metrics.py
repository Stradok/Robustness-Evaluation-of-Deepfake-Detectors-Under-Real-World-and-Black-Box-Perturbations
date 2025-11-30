#!/usr/bin/env python3
"""
quality_metrics.py

Adds PSNR + SSIM to the EXISTING attack_results.csv
without overwriting other columns.
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def compute_psnr_ssim(original, attacked):
    psnr = peak_signal_noise_ratio(original, attacked, data_range=255)
    ssim = structural_similarity(original, attacked, channel_axis=2)
    return psnr, ssim


def main():

    jpeg_dir = Path("attacks_out/jpeg/examples")
    bb_dir = Path("attacks_out/examples")

    results_csv = Path("attacks_out/attack_results.csv")

    # ----------------------------------------------------
    # Load existing attack_results.csv
    # ----------------------------------------------------
    if results_csv.exists():
        print("[INFO] Loading existing attack_results.csv")
        df = pd.read_csv(results_csv)
    else:
        print("[ERROR] attack_results.csv does NOT exist yet. Run attacks first.")
        return

    # Prepare new columns
    df["psnr"] = np.nan
    df["ssim"] = np.nan

    # ----------------------------------------------------
    # Process JPEG attacks
    # ----------------------------------------------------
    print("[INFO] Processing JPEG examples...")

    if jpeg_dir.exists():
        for folder in jpeg_dir.iterdir():  
            attack_name = folder.name  

            for file in folder.iterdir():
                if "orig" in file.name:
                    img_id = file.name.split("_")[0]
                    orig = load_image(file)

                    adv_file = folder / file.name.replace("orig", "jpeg")
                    if adv_file.exists():
                        adv = load_image(adv_file)

                        psnr, ssim = compute_psnr_ssim(orig, adv)

                        df.loc[
                            (df["attack"] == attack_name) &
                            (df["image_id"] == int(img_id)),
                            ["psnr", "ssim"]
                        ] = [psnr, ssim]

    # ----------------------------------------------------
    # Process black-box attacks
    # ----------------------------------------------------
    print("[INFO] Processing black-box attack examples...")

    if bb_dir.exists():
        for folder in bb_dir.iterdir():
            attack_name = folder.name

            for file in folder.iterdir():
                if "orig" in file.name:
                    img_id = file.name.split("_")[0]
                    orig = load_image(file)

                    adv_file = folder / file.name.replace("orig", "adv")
                    if adv_file.exists():
                        adv = load_image(adv_file)
                        psnr, ssim = compute_psnr_ssim(orig, adv)

                        df.loc[
                            (df["attack"] == attack_name) &
                            (df["image_id"] == int(img_id)),
                            ["psnr", "ssim"]
                        ] = [psnr, ssim]

    # ----------------------------------------------------
    # Save merged results
    # ----------------------------------------------------
    print("[INFO] Saving merged attack_results.csv")
    df.to_csv(results_csv, index=False)
    print("[DONE] PSNR + SSIM successfully appended!")


if __name__ == "__main__":
    main()
