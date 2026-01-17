import cv2 as cv
import numpy as np
import os

from .config import WINDOWS_4PTS
from .detector import detect_tags
from .refine_idw_patch import refine_idw_patch
from .utils import safe_imwrite
from .binarize import binarize_patch_dual


def bbox_from_template(ids, layout, img_shape, margin=10):
    H, W = img_shape[:2]
    xs, ys = [], []

    for mid in ids:
        if mid in layout:
            x, y = layout[mid]
            xs.append(x)
            ys.append(y)

    x_min = max(0, int(min(xs)) - margin)
    y_min = max(0, int(min(ys)) - margin)
    x_max = min(W, int(max(xs)) + margin)
    y_max = min(H, int(max(ys)) + margin)

    return x_min, y_min, x_max, y_max


def refine_regions(template_img, layout, warped_src, windows=WINDOWS_4PTS, debug_dir=None):
    base = template_img.copy()
    gray = cv.cvtColor(warped_src, cv.COLOR_BGR2GRAY)
    dets = {d.id: d for d in detect_tags(gray)}

    for wi, ids in enumerate(windows):
        ids = [i for i in ids if i in dets and i in layout]
        if len(ids) < 4:
            continue

        x0, y0, x1, y1 = bbox_from_template(ids, layout, base.shape)

        patch = warped_src[y0:y1, x0:x1]
        off = np.array([x0, y0], np.float32)

        src = np.array([dets[i].center - off for i in ids], np.float32)
        dst = np.array([np.array(layout[i], np.float32) - off for i in ids])

        H, _ = cv.findHomography(src, dst, cv.RANSAC, 3.0)
        if H is None:
            continue

        ph, pw = patch.shape[:2]

        patch_H = cv.warpPerspective(
            patch,
            H,
            (pw, ph),
            borderMode=cv.BORDER_REPLICATE,
        )

        src_local_H = cv.perspectiveTransform(
            src.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        patch_refined = refine_idw_patch(
            patch_H,
            src_local=src_local_H,
            dst_local=dst,
            grid_shape=(12, 12),
            idw_power=3.0,
        )

        base_patch = base[y0:y1, x0:x1]

        base_patch[:] = 255

        mask_ink = binarize_patch_dual(patch_refined)

        base_patch[mask_ink] = 0

        base[y0:y1, x0:x1] = base_patch

        if debug_dir:
            safe_imwrite(
                os.path.join(debug_dir, f"step_region_{wi:02d}.png"),
                base,
            )

    return base
