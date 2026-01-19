"""
STEP 1: BATCH PREPROCESSING (Multi-Scale HSI + WR Mean Calibration)
Save this file as: preprocess_hsi_WR_mean.py

Features:
1. Loads .hdr/.dat HSI cubes.
2. Rotates images 90 degrees to correct orientation.
3. Renames output to standardized frame_00001 format.
4. Extracts TARGET_BANDS and saves full-resolution .npy.
5. Saves downscaled versions (1/2, 1/4, 1/8).
6. (Optional) Saves pseudo-RGB for COLMAP.
7. (NEW) White Reference calibration:
       I_norm(x,y,λ) = I(x,y,λ) / WR_mean_smooth(λ)
   where WR_mean_smooth is the smoothed mean WR spectrum
   computed only from the white-tarp ROI.
"""

import spectral.io.envi as envi
import numpy as np
import os
import glob
import cv2
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAR_DIR = r"nerfstudio/_custom_dataset"
source_list = ['honeyscrisp/NC1_t1', 'maize/Oh43x_SC', 'pear']
SOURCE = source_list[1]      # change if needed

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

# WR tarp ROI (after rotation)
WR_X_MIN, WR_Y_MIN = 150, 150
WR_X_MAX, WR_Y_MAX = 380, 370   # inclusive


# ---------------------------------------------------------
# Helper: smoothing (spectral direction)
# ---------------------------------------------------------
def smooth_1d(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Simple moving-average smoothing along spectral axis."""
    signal = np.asarray(signal, dtype=np.float32).ravel()
    if kernel_size <= 1:
        return signal
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------
# Compute WR mean spectrum
# ---------------------------------------------------------
def compute_wr_mean_spectrum():
    """
    Computes the mean WR spectrum over the tarp ROI, then smooths it.
    Returns:
        wr_mean_smooth: shape (B,)
    """
    print("[WR] Loading WR cube...")
    img_obj = envi.open(WR_HDR_PATH)
    wr_data = img_obj.load()  # (H, W, Bands)

    # Rotate WR image same as plant data
    wr_data = np.rot90(wr_data, k=-1)

    h, w, total_bands = wr_data.shape
    selected_indices = np.linspace(0, total_bands - 1, TARGET_BANDS, dtype=int)
    wr_reduced = wr_data[:, :, selected_indices].astype(np.float32)  # shape (H,W,B)

    # Extract WR tarp ROI
    x0 = max(0, min(WR_X_MIN, w - 1))
    x1 = max(0, min(WR_X_MAX + 1, w))
    y0 = max(0, min(WR_Y_MIN, h - 1))
    y1 = max(0, min(WR_Y_MAX + 1, h))

    wr_roi = wr_reduced[y0:y1, x0:x1, :]   # (H_roi, W_roi, B)

    # Mean over all pixels for each band
    wr_pixels = wr_roi.reshape(-1, wr_roi.shape[2])   # (N, B)
    wr_mean_raw = wr_pixels.mean(axis=0)              # (B,)
    
    
    #----------------------------------------------------------------------------------------START
    # ---------- 1st-pass mean ----------
    wr_mean_raw = wr_pixels.mean(axis=0)  # (B,)

    # ---------- compute per-pixel relative deviation ----------
    dev_cube = wr_roi - wr_mean_raw.reshape(1, 1, -1)           # (H_roi,W_roi,B)
    dev_abs_xy = np.mean(np.abs(dev_cube), axis=2)              # (H_roi,W_roi)
    mean_xy    = np.mean(wr_roi, axis=2)                        # (H_roi,W_roi)
    rel_dev_xy = dev_abs_xy / (mean_xy + 1e-6)                  # avoid /0
    
    print("median rel dev:", np.median(rel_dev_xy))
    print("95% rel dev   :", np.percentile(rel_dev_xy, 95))
    print("max   rel dev  :", rel_dev_xy.max())

    # choose a threshold or percentile for "good" pixels
    # option A: fixed threshold (e.g., 10% relative deviation)
    rel_thr = 0.10
    # option B (alternative): percentile-based
    # rel_thr = np.percentile(rel_dev_xy, 80)   # keep best 80%

    good_mask = rel_dev_xy < rel_thr           # boolean (H_roi,W_roi)
    good_pixels = wr_roi[good_mask]           # (N_good, B)

    print(f"[WR] Using {good_pixels.shape[0]} / {wr_pixels.shape[0]} pixels "
          f"({100*good_pixels.shape[0]/wr_pixels.shape[0]:.1f}%) for WR mean.")

    # ---------- refined mean from central/low-deviation pixels ----------
    wr_mean_refined = good_pixels.mean(axis=0)           # (B,)
    wr_mean_smooth  = np.clip(wr_mean_refined, 1e-4, None)

    # ---------- recompute deviation stats with refined mean (for reporting) ----------
    dev_cube2   = wr_roi - wr_mean_smooth.reshape(1, 1, -1)
    dev_abs_xy2 = np.mean(np.abs(dev_cube2), axis=2)
    mean_xy2    = np.mean(wr_roi, axis=2)
    rel_dev2    = (dev_abs_xy2 / (mean_xy2 + 1e-6)).ravel()

    med2 = np.median(rel_dev2)
    p95_2 = np.percentile(rel_dev2, 95)
    max2 = np.max(rel_dev2)
    print("[WR] After central masking:")
    print(f"    median rel dev : {med2:.4f}")
    print(f"    95% rel dev    : {p95_2:.4f}")
    print(f"    max rel dev    : {max2:.4f}")

    # (optional) regenerate the x–y deviation plot using dev_abs_xy2 or rel_dev2
    #----------------------------------------------------------------------------------------END


    # Smooth the mean spectrum across bands
    # wr_mean_smooth = smooth_1d(wr_mean_raw, kernel_size=3)
    # Avoid division by zero
    wr_mean_smooth = np.clip(wr_mean_raw, 1e-4, None)
    
    
    #----------------------------------------------------------------------------------------START
    # 2D (x-y) intensity deviation map with λ collapsed
    # D(x,y) = mean_λ | I_WR(x,y,λ) - μ_WR(λ) |
    # ---------------------------------------------------------
    import matplotlib.pyplot as plt
    from pathlib import Path

    # deviation cube: (H_roi, W_roi, B)
    dev_cube = wr_roi - wr_mean_smooth.reshape(1, 1, -1)

    # per-pixel mean absolute deviation over λ  -> (H_roi, W_roi)
    dev_xy = np.mean(np.abs(dev_cube), axis=2)

    # per-pixel mean reflectance over λ (for background)
    mean_xy = wr_roi.mean(axis=2)

    # normalize to [0,1] for visualization
    dev_norm = (dev_xy - dev_xy.min()) / (dev_xy.ptp() + 1e-8)
    mean_norm = (mean_xy - mean_xy.min()) / (mean_xy.ptp() + 1e-8)

    base = Path(WR_HDR_PATH).with_suffix("")

    # ---------- Figure 1: pure deviation heatmap ----------
    plt.figure(figsize=(5, 4))
    im = plt.imshow(dev_norm, origin="upper", cmap="magma")
    plt.colorbar(im, label=r"mean$_\lambda\,|I_{WR}(x,y,\lambda)-\mu_{WR}(\lambda)|$")
    plt.title("WR Intensity Deviation (x–y, λ-mean)")
    plt.xlabel("x (within WR ROI)")
    plt.ylabel("y (within WR ROI)")
    plt.tight_layout()
    out1 = str(base) + "_xy_deviation.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"[WR] Saved x–y deviation map → {out1}")

    # ---------- Figure 2: mean + transparent deviation overlay ----------
    plt.figure(figsize=(5, 4))
    # background: mean reflectance
    plt.imshow(mean_norm, origin="upper", cmap="gray")
    # overlay: deviation, alpha ∝ deviation magnitude
    plt.imshow(dev_norm,
               origin="upper",
               cmap="magma",
               alpha=dev_norm * 0.8)
    plt.title("WR Mean Reflectance with Deviation Overlay")
    plt.xlabel("x (within WR ROI)")
    plt.ylabel("y (within WR ROI)")
    plt.tight_layout()
    out2 = str(base) + "_xy_mean_plus_deviation.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"[WR] Saved mean+deviation overlay → {out2}")
    #----------------------------------------------------------------------------------------END

        
    # print(wr_pixels.shape, wr_mean_raw.shape, wr_mean_smooth.shape)

    print("[WR] WR mean raw (first 5):", wr_mean_raw[:5])
    print("[WR] WR mean smooth (first 5):", wr_mean_smooth[:5])

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
            wr_mean_spectrum = compute_wr_mean_spectrum()
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

            # --------------------------
            # WR mean normalization
            # --------------------------
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
            
             # --------------------------
            # <<< NEW >>> Save calibrated pseudo-RGB PNG for COLMAP
            # --------------------------
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

                # Robust scaling for visualization
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
            # --------------------------

            # Save multi-scale npy
            '''
            for scale, folder_name in scales.items():
                save_path = os.path.join(BASE_OUTPUT_DIR, folder_name, f"{frame_name}.npy")
                if scale == 1:
                    final_data = reduced_data
                else:
                    new_w, new_h = int(w / scale), int(h / scale)
                    final_data = cv2.resize(reduced_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

                np.save(save_path, final_data)
            '''

            print(f"[{frame_counter}] {filename} -> {frame_name}")
            frame_counter += 1
            

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("-" * 50)
    print("DONE! You can now run COLMAP + Nerfstudio.")
    print("Remember to copy the .npy folders into the COLMAP directory before training.")


if __name__ == "__main__":
    process_batch()
