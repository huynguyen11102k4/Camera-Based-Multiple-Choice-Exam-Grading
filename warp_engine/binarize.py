import cv2 as cv
import numpy as np


def binarize_patch_dual(patch_bgr, blur_ksize=1, debug_dir=None, region_id=None):
    gray = cv.cvtColor(patch_bgr, cv.COLOR_BGR2GRAY)

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id}_gray.png", gray)

    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        if debug_dir and region_id is not None:
            cv.imwrite(f"{debug_dir}/region_{region_id}_blurred.png", gray)

    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    flat = gray.reshape(-1)

    fill_th = np.percentile(flat, 8)

    mask_fill = gray < fill_th

    kernel = np.ones((1, 1), np.uint8)
    mask_fill = cv.erode(
        mask_fill.astype(np.uint8) * 255,
        kernel,
        iterations=1,
    ) > 0

    mask_ink = mask_fill

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id}_mask.png", mask_ink * 255)

    return mask_ink
