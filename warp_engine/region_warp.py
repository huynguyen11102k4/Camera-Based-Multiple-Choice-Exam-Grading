# path: warp_engine/region_warp.py
import logging
import os

import cv2 as cv
import numpy as np

from .config import WINDOWS_4PTS
from .detector import detect_tags
from .utils import safe_imwrite
from .refine_idw_patch import refine_idw_patch

log = logging.getLogger(__name__)

def bbox_from_template(ids, layout, img_shape, margin=10):
    """
    Tính bounding box trong hệ A4 dựa trên vị trí marker trong template.

    Vì warped đã gần khớp template, nên vị trí layout[] có thể dùng trực tiếp
    để xác định vùng patch cần xử lý local-H + local-IDW.
    """
    H, W = img_shape[:2]
    xs, ys = [], []

    for mid in ids:
        if mid in layout:
            x, y = layout[mid]
            xs.append(x)
            ys.append(y)

    if not xs:
        raise RuntimeError("bbox_from_template(): No valid ids in template layout.")

    x_min = int(np.floor(min(xs))) - margin
    x_max = int(np.ceil(max(xs))) + margin
    y_min = int(np.floor(min(ys))) - margin
    y_max = int(np.ceil(max(ys))) + margin

    # Clamp vào ảnh
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError(
            f"bbox_from_template(): Invalid bbox ({x_min},{y_min})-({x_max},{y_max})"
        )

    return x_min, y_min, x_max, y_max


def refine_regions(warped, layout, windows=WINDOWS_4PTS, debug_dir=None):
    base = warped.copy()
    dets = {d.id: d for d in detect_tags(cv.cvtColor(base, cv.COLOR_BGR2GRAY))}

    for wi, ids in enumerate(windows):
        ids = [i for i in ids if i in dets and i in layout]
        if len(ids) < 4:
            continue

        # --- BBOX theo template ---
        x0, y0, x1, y1 = bbox_from_template(ids, layout, base.shape)

        patch = base[y0:y1, x0:x1]
        off = np.array([x0, y0], np.float32)

        # --- Local coordinates ---
        src = np.array([dets[i].center - off for i in ids], np.float32)
        dst = np.array([np.array(layout[i], np.float32) - off for i in ids])

        # --- Local-H ---
        H, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0)
        if H is None:
            continue

        patch_h, patch_w = patch.shape[:2]

        patch_H = cv.warpPerspective(
            patch,
            H,
            (patch_w, patch_h),
            borderMode=cv.BORDER_REPLICATE,
        )

        # ------------------------------------------------------
        #         Local-IDW sau khi H (để sửa méo phi tuyến)
        # ------------------------------------------------------
        # src_local_H = vị trí sau warp H
        src_local_H = cv.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)

        patch_H_IDW = refine_idw_patch(
            patch_H,
            src_local=src_local_H,
            dst_local=dst,
            grid_shape=(12, 12),
            idw_power=3.0,
        )

        base[y0:y1, x0:x1] = patch_H_IDW

        if debug_dir:
            safe_imwrite(
                os.path.join(debug_dir, f"step4_region_{wi:02d}.png"),
                base,
                note=f"region {wi} after local-H + IDW",
            )

    return base
