import numpy as np
import cv2 as cv
import os
from .detector import detect_tags
from .utils import safe_imwrite, safe_mkdir, draw_detections
import logging

log = logging.getLogger(__name__)


def collect_correspondences(detections, layout, restrict_ids=None):
    src, dst = [], []
    for d in detections:
        if d.id not in layout:
            continue
        if restrict_ids and d.id not in restrict_ids:
            continue
        src.append(d.center.astype(np.float32))
        dst.append(np.array(layout[d.id], np.float32))

    if len(src) < 4:
        raise RuntimeError("Matched markers < 4 for homography")

    return np.array(src), np.array(dst)


def warp_single_h(img, layout, out_size, debug_dir=None):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)

    if debug_dir:
        safe_mkdir(debug_dir)
        vis = draw_detections(img, detections)
        safe_imwrite(os.path.join(debug_dir, "step1_input_tags.png"), vis)

    src, dst = collect_correspondences(detections, layout)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("Global homography failed")

    warped = cv.warpPerspective(img, H, out_size)

    if debug_dir:
        safe_imwrite(os.path.join(debug_dir, "step2_warped_global_H.png"), warped)

    return warped
