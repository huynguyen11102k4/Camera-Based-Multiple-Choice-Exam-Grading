import os
import logging
import numpy as np
import cv2 as cv

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_imwrite(path: str, img: np.ndarray, note: str = ""):
    ok = cv.imwrite(path, img)
    if ok:
        if note:
            log.info(f"[WarpEngine] Saved {note} -> {path}")
        else:
            log.info(f"[WarpEngine] Saved -> {path}")
    else:
        log.warning(f"[WarpEngine] Failed to save image -> {path}")


def draw_detections(img_bgr, detections, draw_ids=True, color=(0, 255, 0)):
    vis = img_bgr.copy()
    for d in detections:
        pts = d.corners.astype(int)
        cv.polylines(vis, [pts], True, color, 2)
        cx, cy = d.center.astype(int)
        cv.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        if draw_ids:
            cv.putText(vis, str(d.id), (cx + 5, cy - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis
