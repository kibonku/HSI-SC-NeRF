"""
STEP 1: BATCH PREPROCESSING (Multi-Scale HSI + WR Mean Calibration)
Save this file as: preprocess_hsi_WR_mean.py

Features:
1. Loads .hdr/.dat HSI cubes.
2. Rotates images 90 degrees to correct orientation.
3. Renames output to standardized frame_00001 format.
4. Extracts TARGET_BANDS and saves full-resolution .npy.
5. Saves downscaled versions (1/2, 1/4, 1/8).   <-- currently commented
6. (Optional) Saves pseudo-RGB for COLMAP.
7. White Reference calibration:
       I_norm(x,y,λ) = I(x,y,λ) / WR_mean_smooth(λ)
   where WR_mean_smooth is the refined mean WR spectrum
   computed only from an automatically selected central WR region.
"""

import spectral.io.envi as envi
import numpy as np
import os
import glob
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAR_DIR = r"nerfstudio/_custom_dataset"
source_list = ['honeyscrisp/NC1_t1', 'maize/Oh43x_SC', 'pear']
SOURCE = source_list[2]      # change if needed

TARGET_BANDS = 204           # if using full bands
RGB_BAND_INDICES = (70, 53, 19)

INPUT_DIR = f"{PAR_DIR}/1.raw/{SOURCE}/REFLECTANCE"
BASE_OUTPUT_DIR = f"{PAR_DIR}/2.pre/{SOURCE}/{TARGET_BANDS}_hsi_WR"
rgb_dir = f"{PAR_DIR}/1.raw/{SOURCE}/pseudo_rgb_WR"

# ---------------------------------------------------------
# WHITE REFERENCE SETTINGS
# ---------------------------------------------------------
USE_WR_CALIB = True

# Path to WR cube (update the filename to your actual WR image)
WR_HDR_PATH = r"nerfstudio/_custom_dataset/1.raw/WR_cubert/REFLECTANCE_386.hdr"

# WR tarp ROI (after rotation) – coarse rectangle that contains the WR board
WR_X_MIN, WR_Y_MIN = 74, 124
WR_X_MAX, WR_Y_MAX = 440, 451   # inclusive


# ---------------------------------------------------------
# Helper: smoothing (spectral direction)  [not used now, but kept]
# ---------------------------------------------------------
def smooth_1d(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Simple moving-average smoothing along spectral axis."""
    signal = np.asarray(signal, dtype=np.float32).ravel()
    if kernel_size <= 1:
        return signal
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------
# Compute WR mean spectrum with automatic polygon mask
# ---------------------------------------------------------
def compute_wr_mean_spectrum(rel_percentile: float = 70.0):
    """
    Computes a refined WR mean spectrum from an automatically selected
    central region of the WR tarp.

    Steps:
      1) Load WR cube and extract coarse rectangular WR ROI.
      2) First-pass mean over all ROI pixels.
      3) Compute per-pixel relative deviation:
            r(x,y) = mean_λ |I - μ_raw| / mean_λ I
      4) Threshold r(x,y) at the given percentile to obtain initial mask.
      5) Morphologically close/open the mask and keep the largest component
         → smooth, central polygon-like region.
      6) Compute mean spectrum using only pixels inside this region.
      7) Save:
           - x-y deviation map with contour overlay
           - histogram of relative deviations (masked region)
      8) Return wr_mean_smooth (1D array, length = B).
    """
    print("[WR] Loading WR cube...")
    img_obj = envi.open(WR_HDR_PATH)
    wr_data = img_obj.load()  # (H, W, Bands)

    # Rotate WR image same as plant data
    wr_data = np.rot90(wr_data, k=-1)

    h, w, total_bands = wr_data.shape
    selected_indices = np.linspace(0, total_bands - 1, TARGET_BANDS, dtype=int)
    wr_reduced = wr_data[:, :, selected_indices].astype(np.float32)  # (H,W,B)

    # Extract coarse WR tarp ROI (inclusive bounds)
    x0 = max(0, min(WR_X_MIN, w - 1))
    x1 = max(0, min(WR_X_MAX + 1, w))
    y0 = max(0, min(WR_Y_MIN, h - 1))
    y1 = max(0, min(WR_Y_MAX + 1, h))

    wr_roi = wr_reduced[y0:y1, x0:x1, :]   # (H_roi, W_roi, B)
    H_roi, W_roi, B = wr_roi.shape
    print(f"[WR] WR ROI shape: {H_roi} x {W_roi} x {B}")

    # -----------------------------------------------------
    # 1) First-pass mean and relative deviation r(x,y)
    # -----------------------------------------------------
    wr_pixels = wr_roi.reshape(-1, B)            # (N,B)
    mu_raw = wr_pixels.mean(axis=0)             # (B,)

    dev_cube = wr_roi - mu_raw.reshape(1, 1, -1)
    dev_abs_xy = np.mean(np.abs(dev_cube), axis=2)   # (H_roi,W_roi)
    mean_xy    = np.mean(wr_roi, axis=2)             # (H_roi,W_roi)
    rel_dev_xy = dev_abs_xy / (mean_xy + 1e-6)

    rel_flat = rel_dev_xy.ravel()
    print("[WR] First-pass relative deviation stats (all ROI pixels):")
    print(f"    median rel dev : {np.median(rel_flat):.4f}")
    print(f"    95% rel dev    : {np.percentile(rel_flat, 95):.4f}")
    print(f"    max rel dev    : {rel_flat.max():.4f}")

    # -----------------------------------------------------
    # 2) Threshold + morphology → automatic central mask
    # -----------------------------------------------------
    # Threshold at given percentile (e.g., 70% → keep best 70% lowest deviations)
    thr = np.percentile(rel_flat, rel_percentile)
    print(f"[WR] Thresholding relative deviation at {rel_percentile:.1f} percentile (thr={thr:.4f})")
    mask0 = (rel_dev_xy <= thr).astype(np.uint8)

    # Morphological close/open to smooth the mask
    k_size = max(5, min(H_roi, W_roi) // 10)   # adaptive kernel size
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask_morph = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component (central blob)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph, connectivity=8)
    if num_labels <= 1:
        print("[WR] WARNING: only background found in mask; falling back to mask0.")
        final_mask = mask0.astype(bool)
    else:
        # stats[0] is background; choose largest area label>=1
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + np.argmax(areas)
        final_mask = (labels == best_label)
        print(f"[WR] Connected components: {num_labels-1}, using label {best_label} with area {areas.max()}")

    num_good = int(final_mask.sum())
    frac_good = num_good / wr_pixels.shape[0]
    print(f"[WR] Using {num_good} / {wr_pixels.shape[0]} pixels "
          f"({100*frac_good:.1f}%) inside automatic central mask for WR mean.")

    good_pixels = wr_roi[final_mask]    # (N_good,B)
    
    mu_refined = good_pixels.mean(axis=0)        # (B,)

    # -----------------------------------------------------
    # 3) Recompute deviation with refined mean (for reporting)
    # -----------------------------------------------------
    dev_cube2   = wr_roi - mu_refined.reshape(1, 1, -1)
    dev_abs_xy2 = np.mean(np.abs(dev_cube2), axis=2)
    mean_xy2    = np.mean(wr_roi, axis=2)
    rel_dev2    = dev_abs_xy2 / (mean_xy2 + 1e-6)
    
    rel_all = rel_dev2.ravel()
    rel_good = rel_dev2[final_mask]

    print("\n[WR] After refined mean (all ROI pixels):")
    print(f"    median rel dev : {np.median(rel_all):.4f}")
    print(f"    95% rel dev    : {np.percentile(rel_all, 95):.4f}")
    print(f"    max rel dev    : {np.max(rel_all):.4f}")

    print("\n[WR] After refined mean (automatic central region only):")
    print(f"    median rel dev : {np.median(rel_good):.4f}")
    print(f"    95% rel dev    : {np.percentile(rel_good, 95):.4f}")
    print(f"    max rel dev    : {np.max(rel_good):.4f}")
    
    
    # -----------------------------------------------------
    # 3) Refined mean from automatic polygon region
    # -----------------------------------------------------
    mu_refined = rel_good .mean(axis=0)  # (B,)
    # Smooth the refined WR spectrum
    mu_smoothed = smooth_1d(mu_refined, kernel_size=5)
    # Avoid zero division
    wr_mean_smooth = np.clip(mu_smoothed, 1e-4, None)

    # -----------------------------------------------------
    # 5) Save deviation map + contour + histogram
    # -----------------------------------------------------
    base = Path(WR_HDR_PATH).with_suffix("")
    
        # -----------------------------------------------------
    # Save final mask as FULL-SIZE mask only
    # -----------------------------------------------------
    pseudo_files = sorted(
        f for f in glob.glob(os.path.join(rgb_dir, "*.png"))
        if "WR" not in os.path.basename(f)
    )

    if len(pseudo_files) == 0:
        print("[WR] WARNING: No pseudo-RGB images found; cannot save full-size mask.")
    else:
        # use the first pseudo-RGB to determine full image size
        sample_rgb = cv2.imread(pseudo_files[0])
        H_full, W_full = sample_rgb.shape[:2]

        # create full-size binary mask (all zeros)
        mask_full = np.zeros((H_full, W_full), dtype=np.uint8)

        # place the WR ROI mask in correct position
        # final_mask → shape (H_roi, W_roi)
        mask_full[y0:y1, x0:x1] = (final_mask.astype(np.uint8) * 255)

        # save full-size mask
        mask_path = str(base) + "_WR_mask_full.png"
        cv2.imwrite(mask_path, mask_full)
        print(f"[WR] Saved FULL-SIZE WR mask → {mask_path}")
    

    # Normalize deviation map for visualization
    dev_norm = (dev_abs_xy2 - dev_abs_xy2.min()) / (dev_abs_xy2.ptp() + 1e-8)

    plt.figure(figsize=(6.5, 5))
    im = plt.imshow(dev_norm, origin="upper", cmap="magma")
    cbar = plt.colorbar(im)
    cbar.set_label(r"mean$_\lambda\,|I_{WR}(x,y,\lambda) - \mu_{WR}(\lambda)|$")
    
    plt.title("WR Intensity Deviation (x-y, λ-mean)")
    plt.xlabel("x (within WR ROI)")
    plt.ylabel("y (within WR ROI)")
    plt.tight_layout()
    
    out_xy = str(base) + "_WR_xy_deviation.png"
    plt.savefig(out_xy, dpi=200)
    
    # overlay contour of automatic mask
    plt.contour(final_mask.astype(float), levels=[0.5], colors="red", linewidths=2)
    
    plt.title("WR Intensity Deviation (x-y, λ-mean) with automatic mask")
    out_xy = str(base) + "_WR_xy_deviation_autoMask.png"
    plt.savefig(out_xy, dpi=200)
    plt.close()
    print(f"[WR] Saved x-y deviation map with mask → {out_xy}")

    plt.figure(figsize=(5, 4))
    plt.hist(rel_good.ravel(), bins=50)
    plt.xlabel(r"relative deviation  mean$_\lambda\,|I-\mu|$/mean$_\lambda I$")
    plt.ylabel("Pixel count")
    plt.title("WR relative deviations (automatic central region)")
    plt.tight_layout()
    out_hist = str(base) + "_WR_rel_deviation_hist_autoMask.png"
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[WR] Saved deviation histogram → {out_hist}\n")

    return wr_mean_smooth


# ---------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------
def process_batch():
    # Create output folders
    scales = {1: "images", 2: "images_2", 4: "images_4", 8: "images_8"}
    for scale, folder_name in scales.items():
        Path(os.path.join(BASE_OUTPUT_DIR, folder_name)).mkdir(parents=True, exist_ok=True)

    Path(rgb_dir).mkdir(parents=True, exist_ok=True)

    # Find HSI files
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.hdr")))
    if not files:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dat")))

    print(f"Found {len(files)} HSI cubes (including WR).")
    print(f"Output base: {BASE_OUTPUT_DIR}")
    print("-" * 50)

    # Compute WR mean spectrum
    wr_mean_spectrum = None
    if USE_WR_CALIB:
        try:
            wr_mean_spectrum = compute_wr_mean_spectrum(rel_percentile=70.0)
        except Exception as e:
            print(f"[WR] Error computing WR mean spectrum: {e}")
            wr_mean_spectrum = None

    # Process each cube
    frame_counter = 1
    for filepath in files:
        break
    
        filename = os.path.basename(filepath)

        # Skip the WR file itself
        if USE_WR_CALIB and os.path.abspath(filepath) == os.path.abspath(WR_HDR_PATH):
            print(f"[SKIP] WR cube: {filename}")
            continue

        frame_name = f"frame_{frame_counter:05d}"

        try:
            # Load and rotate
            img_obj = envi.open(filepath)
            data = img_obj.load()
            data = np.rot90(data, k=-1)

            h, w, total_bands = data.shape
            selected_indices = np.linspace(0, total_bands - 1, TARGET_BANDS, dtype=int)
            reduced_data = data[:, :, selected_indices].astype(np.float32)

            # WR mean normalization
            if wr_mean_spectrum is not None:
                if wr_mean_spectrum.shape[0] == reduced_data.shape[2]:
                    reduced_data = reduced_data / wr_mean_spectrum.reshape(1, 1, -1)
                else:
                    print(
                        f"[WR] WARNING: WR spectrum ({wr_mean_spectrum.shape[0]}) "
                        f"!= cube bands ({reduced_data.shape[2]}), skipping WR calibration."
                    )

            # Clip final normalized values
            reduced_data = np.clip(reduced_data, 0.0, 1.0)

            # Save calibrated pseudo-RGB PNG for COLMAP
            B = reduced_data.shape[2]
            try:
                if RGB_BAND_INDICES is not None and max(RGB_BAND_INDICES) < B:
                    r = reduced_data[:, :, RGB_BAND_INDICES[0]]
                    g = reduced_data[:, :, RGB_BAND_INDICES[1]]
                    b = reduced_data[:, :, RGB_BAND_INDICES[2]]
                    rgb_stack = np.dstack([r, g, b]).astype(np.float32)
                elif B == 3:
                    rgb_stack = reduced_data.astype(np.float32)
                else:
                    raise ValueError(
                        f"RGB_BAND_INDICES out of range for {B} bands "
                        f"and B != 3, cannot form RGB."
                    )

                p_min = np.percentile(rgb_stack, 1)
                p_max = np.percentile(rgb_stack, 99)
                if p_max > p_min:
                    rgb_norm = (rgb_stack - p_min) / (p_max - p_min)
                else:
                    rgb_norm = rgb_stack

                rgb_norm = np.clip(rgb_norm, 0.0, 1.0)
                rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

                img = Image.fromarray(rgb_uint8)
                img.save(os.path.join(rgb_dir, f"{frame_name}.png"))
            except Exception as e:
                print(f"[RGB] Failed to save pseudo-RGB for {filename}: {e}")

            # (Optional) save multi-scale .npy – currently commented
            """
            for scale, folder_name in scales.items():
                save_path = os.path.join(BASE_OUTPUT_DIR, folder_name, f"{frame_name}.npy")
                if scale == 1:
                    final_data = reduced_data
                else:
                    new_w, new_h = int(w / scale), int(h / scale)
                    final_data = cv2.resize(reduced_data, (new_w, new_h),
                                            interpolation=cv2.INTER_AREA)
                np.save(save_path, final_data)
            """

            print(f"[{frame_counter}] {filename} -> {frame_name}")
            frame_counter += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("-" * 50)
    print("DONE! You can now run COLMAP + Nerfstudio.")
    print("Remember to copy the .npy folders into the COLMAP directory before training.")


if __name__ == "__main__":
    process_batch()
