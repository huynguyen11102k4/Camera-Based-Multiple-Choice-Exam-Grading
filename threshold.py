# threshold.py
from __future__ import annotations

from typing import Literal

import cv2 as cv
import numpy as np

from core import log, safe_mkdir

ThresholdMethod = Literal[
    "adaptive_gaussian",
    "adaptive_mean",
    "otsu",
]


def threshold_adaptive_gaussian(gray: np.ndarray) -> np.ndarray:
    """Gaussian blur nhẹ rồi adaptive Gaussian threshold.

    Tôi blur 3x3 để giảm noise nhưng vẫn giữ cạnh rõ cho vùng tô.
    """
    log.debug(f"[Thresh] (adaptive_gaussian) gray_shape={gray.shape}, dtype={gray.dtype}")
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    th = cv.adaptiveThreshold(
        blur,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        33,  # blockSize
        5,   # C
    )
    white = int(np.sum(th > 0))
    total = th.size
    log.debug(f"[Thresh] white={white}/{total} ({white/total:.2%})")
    return th


def threshold_adaptive_mean(gray: np.ndarray) -> np.ndarray:
    """Adaptive mean threshold.

    Tôi thêm option này để bạn benchmark với Gaussian.
    """
    log.debug(f"[Thresh] (adaptive_mean) gray_shape={gray.shape}, dtype={gray.dtype}")
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    th = cv.adaptiveThreshold(
        blur,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        33,
        5,
    )
    white = int(np.sum(th > 0))
    total = th.size
    log.debug(f"[Thresh] white={white}/{total} ({white/total:.2%})")
    return th


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """Global Otsu threshold.

    Tôi dùng Otsu làm baseline global threshold để so sánh với adaptive.
    """
    log.debug(f"[Thresh] (otsu) gray_shape={gray.shape}, dtype={gray.dtype}")
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _t, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    white = int(np.sum(th > 0))
    total = th.size
    log.debug(f"[Thresh] white={white}/{total} ({white/total:.2%})")
    return th


def build_threshold(
    gray: np.ndarray,
    method: ThresholdMethod = "adaptive_gaussian",
) -> np.ndarray:
    """Factory chọn phương pháp threshold.

    Tôi dùng factory để bạn đổi thuật toán ở một chỗ, thay vì sửa khắp nơi.
    """
    if method == "adaptive_gaussian":
        return threshold_adaptive_gaussian(gray)
    if method == "adaptive_mean":
        return threshold_adaptive_mean(gray)
    if method == "otsu":
        return threshold_otsu(gray)
    raise ValueError(f"Unsupported threshold method: {method}")


if __name__ == "__main__":
    # Demo threshold: đọc samples/input.jpg, áp dụng 3 kiểu threshold và lưu kết quả.
    input_path = "samples/input.jpg"
    out_dir = "debug_threshold"

    safe_mkdir(out_dir)

    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    for method in ["adaptive_gaussian", "adaptive_mean", "otsu"]:
        th = build_threshold(img, method=method)  # vì muốn so sánh từng method
        out_file = f"{out_dir}/th_{method}.png"
        cv.imwrite(out_file, th)
        log.info(f"[threshold main] Saved {out_file}")
