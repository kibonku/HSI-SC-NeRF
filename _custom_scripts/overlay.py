import cv2
import numpy as np
from pathlib import Path

# -----------------------------------------------------
# USER INPUT
# -----------------------------------------------------
pseudo_rgb_path = r"nerfstudio/_custom_dataset/1.raw/maize/Oh43x_SC/REFLECTANCE/REFLECTANCE_322.png"        # your pseudo-RGB image
mask_path       = r"nerfstudio/_custom_dataset/1.raw/WR_cubert/REFLECTANCE_386_WR_mask_full.png"
output_path     = r"nerfstudio/_custom_dataset/1.raw/WR_cubert/overlay_to_maize.png"

ALPHA = 0.40    # transparency of the mask

# -----------------------------------------------------
# LOAD IMAGES
# -----------------------------------------------------
rgb = cv2.imread(pseudo_rgb_path)
if rgb is None:
    raise FileNotFoundError(f"Could not load pseudo-RGB at {pseudo_rgb_path}")

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Could not load mask at {mask_path}")

h1, w1 = rgb.shape[:2]
h2, w2 = mask.shape[:2]

if (h1 != h2) or (w1 != w2):
    raise ValueError(
        f"Mask size {mask.shape} does not match image size {rgb.shape}"
    )

# -----------------------------------------------------
# CREATE RED MASK OVERLAY
# -----------------------------------------------------
# create copy to draw on
overlay = rgb.copy()

# boolean mask where WR mask is present
mask_bool = mask > 0

# target red color for the mask
red = np.array([0, 0, 255], dtype=np.float32)

# blend only on masked pixels:
# overlay = (1-ALPHA)*original + ALPHA*red
overlay[mask_bool] = (
    (1.0 - ALPHA) * overlay[mask_bool].astype(np.float32)
    + ALPHA * red
).astype(np.uint8)

# -----------------------------------------------------
# SAVE & SHOW
# -----------------------------------------------------
cv2.imwrite(output_path, overlay)
print(f"[INFO] Saved overlay → {output_path}")

# # (optional) show
# cv2.imshow("Overlay", overlay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
