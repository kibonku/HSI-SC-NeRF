
"""
STEP 4: UPDATE TRANSFORMS.JSON (Optional but Recommended)
Save this as 'update_transforms.py'

Run this script AFTER running ns-process-data.
It modifies 'transforms.json' to point explicitly to your .npy files.

Why?
This removes ambiguity. Nerfstudio will know exactly that the source
data is .npy.
"""

import json
import os

# Path to your processed project folder containing transforms.json
RGB_DIR = r"nerfstudio/_custom_dataset/2.pre_final/maize/pseudo_rgb_WR_mask_colmap" 
HSI_DIR = r"nerfstudio/_custom_dataset/2.pre_final/maize/204_hsi_WR_mask" 

RGB_JSON_PATH = os.path.join(RGB_DIR, "transforms.json")
HSI_JSON_PATH = os.path.join(HSI_DIR, "transforms.json")

def update_json():
    if not os.path.exists(RGB_JSON_PATH):
        print(f"Error: Could not find {RGB_JSON_PATH}")
        return

    print(f"Loading {RGB_JSON_PATH}...")
    with open(RGB_JSON_PATH, 'r') as f:
        data = json.load(f)

    frames = data.get("frames", [])
    count = 0
    
    for frame in frames:
        old_path = frame["file_path"]
        # Replace .png extension with .npy
        # e.g. "images/frame_00001.png" -> "images/frame_00001.npy"
        if old_path.endswith(".png"):
            new_path = old_path.replace(".png", ".npy")
            frame["file_path"] = new_path
            count += 1
            
    print(f"Updated {count} frame paths from .png to .npy")

    # Save back
    with open(HSI_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("Saved updated transforms.json")
    print("Now you can run training: ns-train nerfacto --data ./3_hsi")

if __name__ == "__main__":
    update_json()
