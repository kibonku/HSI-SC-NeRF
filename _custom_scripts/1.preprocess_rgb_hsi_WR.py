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

# WR tarp ROI (after rotation)
WR_X_MIN, WR_Y_MIN = 74, 124
WR_X_MAX, WR_Y_MAX = 440, 451   # inclusive


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

    # Smooth the mean spectrum across bands
    # wr_mean_smooth = smooth_1d(wr_mean_raw, kernel_size=3)
    # Avoid division by zero
    wr_mean_smooth = np.clip(wr_mean_raw, 1e-4, None)
    
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
