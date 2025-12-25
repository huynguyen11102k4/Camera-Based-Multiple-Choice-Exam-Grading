import json
import os
import cv2 as cv
from .detector import detect_tags
from .utils import safe_imwrite, draw_detections, safe_mkdir
from .config import TEMPLATE_LAYOUT_FILE
import logging

log = logging.getLogger(__name__)


def extract_template(path_img, path_out=TEMPLATE_LAYOUT_FILE, debug_dir=None):
    img = cv.imread(path_img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    detections = detect_tags(gray)

    if debug_dir:
        safe_mkdir(debug_dir)
        vis = draw_detections(img, detections)
        safe_imwrite(os.path.join(debug_dir, "layout_debug.png"), vis)

    layout = {}
    for d in detections:
        layout[d.id] = d.center.tolist()

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    return layout


def load_template(path=TEMPLATE_LAYOUT_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
