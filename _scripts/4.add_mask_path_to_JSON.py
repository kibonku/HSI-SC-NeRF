import json
from pathlib import Path

# === 1. Nerfstudio dataset root ===
DATA_DIR = Path("{MASK_DIR}")  # Update this to your actual dataset path containing transforms.json and the masks folder

TF_PATH = DATA_DIR / "transforms.json"
MASK_REL_DIR = "masks"   # transforms.json will point to masks using this relative path; e.g. "masks/frame_00001.png" for each frame's mask_path

# === 2. folder with mask files ===
MASK_DIR = DATA_DIR / MASK_REL_DIR   # upload masks to this folder before running the script 

assert TF_PATH.exists(), f"transforms.json not found: {TF_PATH}"
assert MASK_DIR.exists(), f"Mask dir not found: {MASK_DIR}"

with open(TF_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

frames = meta["frames"]
print(f"#frames in transforms.json: {len(frames)}")

missing = 0
for i, frame in enumerate(frames):
    fp = Path(frame["file_path"])      # e.g. images/frame_00001.npy
    stem = fp.stem                     # frame_00001

    mask_filename = stem + ".png"
    mask_path_rel = f"{MASK_REL_DIR}/{mask_filename}"     # e.g. masks/frame_00001.png
    mask_path_abs = MASK_DIR / mask_filename

    if not mask_path_abs.exists():
        print(f"[WARN] mask not found for {fp} -> {mask_path_abs}")
        missing += 1

    frame["mask_path"] = mask_path_rel

print(f"Total missing masks: {missing}")

with open(TF_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=4)

print("✅ transforms.json updated with mask_path for all frames.")
