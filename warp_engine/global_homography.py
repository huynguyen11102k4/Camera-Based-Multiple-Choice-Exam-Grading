import numpy as np
import cv2 as cv
from .detector import detect_tags

def collect_correspondences(detections, layout):
    src, dst = [], []
    for d in detections:
        if d.id in layout:
            src.append(d.center.astype(np.float32))
            dst.append(np.array(layout[d.id], np.float32))

    if len(src) < 4:
        raise RuntimeError("Matched markers < 4 for homography")

    return np.array(src), np.array(dst)


def compute_global_h(img, layout):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)

    src, dst = collect_correspondences(detections, layout)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("Global homography failed")

    return H, detections
