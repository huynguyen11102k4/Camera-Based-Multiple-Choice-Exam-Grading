import cv2 as cv
import numpy as np


def warpRegion(img, src_corners, dst_width, dst_height):
    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(src_corners, dst)
    warped = cv.warpPerspective(img, M, (dst_width, dst_height))
    return warped, M