import cv2
import numpy as np
from pathlib import Path

# ================== 사용자 설정 ==================
SRC_DIR = Path(r"nerfstudio/_custom_dataset/1.raw/honeyscrisp/NC1_t1/pseudo_rgb")  # 원본 이미지 폴더
OUT_DIR = Path(r"nerfstudio/_custom_dataset/1.raw/honeyscrisp/NC1_t1/pseudo_rgb_cal")  # 저장할 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 밝기/대비 조절 값 (여기만 바꿔가면서 실험)
ALPHA = 1.5   # contrast (1.0 = 그대로, 1.2~1.6 정도 권장)
BETA  = 10    # brightness (+10 ~ +30 정도 권장)
# =================================================

def adjust_image(img, alpha=1.5, beta=10):
    """
    img: BGR uint8 이미지
    alpha: contrast scale
    beta: brightness shift
    """
    # convertScaleAbs: new_img = saturate(alpha * img + beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def main():
    img_paths = sorted(
        list(SRC_DIR.glob("*.png")) +
        list(SRC_DIR.glob("*.jpg")) +
        list(SRC_DIR.glob("*.jpeg"))
    )

    print(f"[INFO] Found {len(img_paths)} images.")

    for i, p in enumerate(img_paths, 1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read {p}, skip.")
            continue

        adjusted = adjust_image(img, ALPHA, BETA)

        out_path = OUT_DIR / p.name
        cv2.imwrite(str(out_path), adjusted)

        if i % 10 == 0 or i == 1 or i == len(img_paths):
            print(f"[INFO] Processed {i}/{len(img_paths)}: {p.name}")

    print("[DONE] All images processed.")

if __name__ == "__main__":
    main()
