from __future__ import annotations

import os

import cv2 as cv

from warp_engine.config import TEMPLATE_LAYOUT_FILE, A4_PX
from warp_engine.engine import WarpEngine
from warp_engine.template import extract_template
from warp_engine.utils import safe_mkdir, safe_imwrite

INPUT_IMAGE = "samples/1photo2.jpg"              # ảnh chụp cần warp
TEMPLATE_IMAGE = "samples/template_scan1.png"    # ảnh template để extract layout
OUT_DIR = "debug_markers"  # thư mục output

OUTPUT_SIZE = A4_PX                              # (2481, 3509)
USE_GLOBAL_IDW = True
USE_REGION_REFINE = True

USE_EXISTING_TEMPLATE = False


def main():

    safe_mkdir(OUT_DIR)

    if not USE_EXISTING_TEMPLATE:
        print(f"[WarpEngine] Extracting template từ ảnh: {TEMPLATE_IMAGE}")
        extract_template(TEMPLATE_IMAGE, TEMPLATE_LAYOUT_FILE, debug_dir=OUT_DIR)
    else:
        if not os.path.exists(TEMPLATE_LAYOUT_FILE):
            raise FileNotFoundError(
                f"Không tìm thấy {TEMPLATE_LAYOUT_FILE}. "
                f"Bạn cần đặt USE_EXISTING_TEMPLATE=False để extract."
            )

    print(f"[WarpEngine] Loading layout từ {TEMPLATE_LAYOUT_FILE}")
    warp_engine = WarpEngine(TEMPLATE_LAYOUT_FILE)

    img = cv.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh input: {INPUT_IMAGE}")

    print(f"[WarpEngine] Input loaded: shape={img.shape}")

    warped = warp_engine.warp(
        img,
        out_size=OUTPUT_SIZE,
        debug_dir=OUT_DIR,
        use_global_idw=USE_GLOBAL_IDW,
        use_region_refine=USE_REGION_REFINE,
    )

    out_path = os.path.join(OUT_DIR, "warped_a4.png")
    safe_imwrite(out_path, warped, "Warped A4")

    print("\n================= DONE =================")
    print(f"Warped output  →  {out_path}")
    print(f"Debug images   →  {OUT_DIR}")
    print("========================================\n")


if __name__ == "__main__":
    main()
