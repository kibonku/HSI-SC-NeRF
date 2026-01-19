
"""
STEP 1: BATCH PREPROCESSING (Multi-Scale HSI + Pseudo-RGB)
Save this as 'preprocess_hsi.py'

Features:
1. Scans .hdr / .dat files.
2. ROTATES images 90 deg (Fix raw orientation).
3. RENAMES files to sequential 'frame_00001' format.
4. Saves Full-Res .npy (10 bands) in 'images'.
5. Saves Downscaled .npy (1/2, 1/4, 1/8) in 'images_2', 'images_4', etc.
6. Saves Pseudo-RGB .png for COLMAP.
"""

import spectral.io.envi as envi
import numpy as np
import os
import glob
import cv2
from pathlib import Path
from PIL import Image

# --- CONFIGURATION ---
PAR_DIR = r"nerfstudio/_custom_dataset"  
SOURCE = "maize/Oh43x_SC"   #-- Replace --#

TARGET_BANDS = 204  # Actual band=204   #-- Replace --#
# RGB_BAND_INDICES = (70, 53, 19) 

INPUT_DIR = f"{PAR_DIR}/1.raw/{SOURCE}/REFLECTANCE"  
BASE_OUTPUT_DIR = f"{PAR_DIR}/2.pre/{SOURCE}/{TARGET_BANDS}_hsi"
rgb_dir = f"{PAR_DIR}/1.raw/{SOURCE}/pseudo_rgb"

def process_batch():
    # 1. Setup Directories
    scales = {1: "images", 2: "images_2", 4: "images_4", 8: "images_8"}
    for scale, folder_name in scales.items():
        Path(os.path.join(BASE_OUTPUT_DIR, folder_name)).mkdir(parents=True, exist_ok=True)
    
    # Pseudo-RGB goes into a separate folder initially to run colmap on
    Path(rgb_dir).mkdir(parents=True, exist_ok=True)
    
    # Find files
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.hdr")))
    if not files:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dat")))
    
    print(f"Found {len(files)} HSI cubes.")
    print(f"Output Base: {BASE_OUTPUT_DIR}")
    print("-" * 50)
    
    ### CHECK POINT ###
    # print('files:', files)
    # files = [r'nerfstudio/_custom_dataset/1.raw/maize/Oh43x_SC/REFLECTANCE/REFLECTANCE_366.hdr']
    
    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        
        # Standardized naming: frame_00001, frame_00002...
        frame_name = f"frame_{idx+1:05d}"
        
        try:
            # --- LOAD & ROTATE ---
            img_obj = envi.open(filepath)
            data = img_obj.load() # (H, W, Bands)
            
            # Rotate 90 deg Clockwise (fix raw orientation)
            data = np.rot90(data, k=-1)
            
            # --- EXTRACT TARGET BANDS ---
            h, w, total_bands = data.shape
            selected_indices = np.linspace(0, total_bands - 1, TARGET_BANDS, dtype=int)
            reduced_data = data[:, :, selected_indices]
            
            reduced_data = reduced_data.astype(np.float32)
            # Check if likely already reflectance (0-1)
            # Use 99.9th percentile to be robust against hot pixel outliers
            robust_max = np.percentile(reduced_data, 99.9)
            if robust_max > 1.05: 
                # Data is likely in DN (e.g. 0-65535) or Radiance. Normalize it.
                print(f"  > Normalizing {filename}: Max={robust_max:.2f} -> 1.0")
                if robust_max > 0: reduced_data /= robust_max
            else:
                # Data is likely already reflectance. Just clip to be safe.
                pass
            
            # Clip (Float32)
            reduced_data = np.clip(reduced_data, 0.0, 1.0)

            # --- SAVE MULTI-SCALE NPY ---
            for scale, folder_name in scales.items():
                # SAVE AS standardized name
                save_path = os.path.join(BASE_OUTPUT_DIR, folder_name, f"{frame_name}.npy")
                
                if scale == 1:
                    final_data = reduced_data
                else:
                    # Downscale
                    new_w, new_h = int(w / scale), int(h / scale)
                    final_data = cv2.resize(reduced_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                np.save(save_path, final_data)

            # --- SAVE PSEUDO-RGB PNG (for COLMAP) ---
            """
            r = data[:, :, RGB_BAND_INDICES[0]]
            g = data[:, :, RGB_BAND_INDICES[1]]
            b = data[:, :, RGB_BAND_INDICES[2]]
            rgb_stack = np.dstack((r, g, b))
            rgb_stack = rgb_stack.astype(np.float32)
            
            # Robust RGB visualization
            rgb_max = np.percentile(rgb_stack, 99.9)
            rgb_stack /= (rgb_max + 1e-6)
            
            rgb_stack = np.clip(rgb_stack * 255, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(rgb_stack)
            # SAVE AS standardized name
            img.save(os.path.join(rgb_dir, f"{frame_name}.png"))
            """            
            
            print(f"[{idx+1}/{len(files)}] {filename} -> {frame_name}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("-" * 50)
    print("DONE. DATA PREPARATION WORKFLOW:")
    print("1. Run COLMAP on Pseudo-RGB:")
    print(f"   ns-process-data images --data {rgb_dir} --output-dir ./3_hsi")
    print("")
    print("2. [CRITICAL] COPY NPY FOLDERS:")
    print("   'ns-process-data' creates 'images', 'images_2', etc. inside './3_hsi' containing PNGs.")
    print("   You must COPY/OVERWRITE them with your .npy folders generated here:")
    print(f"   cp -r {os.path.join(BASE_OUTPUT_DIR, 'images*')} ./3_hsi/")
    print("   (This puts .npy files right next to the .png files at every scale level)")
    print("")
    print("3. Train:")
    print("    ns-train nerfacto --data ./3_hsi --pipeline.model.num-output-channels 10 ...")
 
if __name__ == "__main__":
    process_batch()
