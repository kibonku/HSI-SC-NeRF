from pathlib import Path
import json

# === 1. Nerfstudio 데이터셋 루트 ===
DATA_DIR = Path(r"nerfstudio/_custom_dataset/2.pre_final/pear/204_hsi_WR_mask")  # <- 네 transforms.json 있는 폴더로 수정

TF_PATH = DATA_DIR / "transforms.json"
MASK_REL_DIR = "masks"   # transforms.json에서 쓸 상대 경로 이름

# === 2. 마스크 파일이 들어있는 폴더 ===
MASK_DIR = DATA_DIR / MASK_REL_DIR   # 이미 frame_00001.png 등 복사해두었다고 가정

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
