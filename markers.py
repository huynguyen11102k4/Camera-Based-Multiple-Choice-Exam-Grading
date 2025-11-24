# markers.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

from core import ARUCO_DICT, A4_PX, TEMPLATE_LAYOUT_FILE, log, safe_mkdir, APRILTAG_FAMILY, TAG_SYSTEM


def build_detector():
    """Khởi tạo detector thống nhất ArUco hoặc AprilTag.

    Tôi tách hàm này để backend có thể thay đổi (aruco / apriltag)
    mà không ảnh hưởng các hàm phía sau.
    """
    if TAG_SYSTEM == "aruco":
        # --- Original ArUco ---
        params = cv.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
        return ("aruco", cv.aruco.ArucoDetector(dictionary, params))

    else:
        # --- AprilTag Detector ---
        options = cv.aruco.AprilTagDetector_Params()
        # Bạn có thể chỉnh refine giống ArUco:
        options.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

        detector = cv.aruco.AprilTagDetector()
        detector.addFamily(APRILTAG_FAMILY, options)

        return ("apriltag", detector)


# ========== ArUco Helpers ==========
def build_aruco_detector() -> cv.aruco.ArucoDetector:
    """Khởi tạo detector với tinh chỉnh subpixel để ổn định góc.

    Tôi dùng SUBPIX vì nó giảm jitter vị trí góc marker, từ đó homography ổn định hơn.
    """
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv.aruco.ArucoDetector(dictionary, params)


def detect_markers(gray: np.ndarray):
    """Phát hiện marker ArUco hoặc AprilTag tùy cấu hình.

    Tôi unified output về cùng chuẩn:
    - corners: List[np.ndarray] mỗi cái (1, 4, 2)
    - ids: np.ndarray shape (N, 1)
    """
    mode, detector = build_detector()

    if mode == "aruco":
        corners, ids, _rej = detector.detectMarkers(gray)
        if ids is None:
            return [], None
        return list(corners), ids

    else:
        # AprilTag detection
        results = detector.detect(gray)
        # results: list of cv2.aruco.AprilTagDetection
        if len(results) == 0:
            return [], None

        corners = []
        ids = []
        for r in results:
            # r.corners shape (4,2) → convert thành (1,4,2) giống ArUco
            c = np.array(r.corners, dtype=np.float32).reshape(1, 4, 2)
            corners.append(c)
            ids.append([r.id])

        ids = np.array(ids, dtype=np.int32)
        return corners, ids


# ========== Template Layout I/O ==========
def extract_template_marker_layout(
    img_path: str,
    save_path: str = TEMPLATE_LAYOUT_FILE,
) -> None:
    """Tạo layout mốc từ ảnh template, lưu {marker_id: [cx, cy]}.

    Tôi lưu tâm marker (thay vì 4 góc) vì fit homography với tâm đơn giản hơn.
    """
    import json

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {img_path}")
    log.debug(f"[Template] Input: {img_path}, shape={img.shape}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(f"[Template] Detected ids: {None if ids is None else ids.flatten().tolist()}")

    if ids is None or len(ids) < 4:
        raise RuntimeError("Template must contain ≥4 ArUco markers.")

    layout: Dict[int, List[float]] = {}
    for i, mid in enumerate(ids.flatten()):
        center = np.mean(corners[i][0], axis=0).tolist()  # trung bình 4 đỉnh
        mid_int = int(mid)
        log.debug(f"[Template] id={mid_int} center={center}")
        layout[mid_int] = center

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    log.info(f"[Template] Saved {len(layout)} markers -> {save_path}")


def load_template_marker_layout(path: str = TEMPLATE_LAYOUT_FILE) -> Dict[int, List[float]]:
    """Đọc layout mốc {id: [cx, cy]} đã lưu từ template.

    Tôi ép kiểu về int/float để tránh lỗi parse lặt vặt từ JSON.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layout = {int(k): [float(x) for x in v] for k, v in data.items()}
    log.debug(f"[Template] loaded layout ids={sorted(layout.keys())}")
    return layout


# ========== Warp & Crop ==========
def warp_to_a4(
    img: np.ndarray,
    template_layout: Dict[int, List[float]],
    out_size: Tuple[int, int] = A4_PX,
) -> np.ndarray:
    """Ước lượng homography từ vị trí marker hiện tại sang layout chuẩn A4.

    Tôi dùng homography dựa trên tâm marker vì cách này ít phụ thuộc orientation của marker.
    """
    log.debug(f"[Warp] input_shape={img.shape}, out_size={out_size}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(
        f"[Warp] detected_ids={None if ids is None else ids.flatten().tolist()}"
    )

    if ids is None or len(ids) < 4:
        raise RuntimeError("Not enough markers to warp to A4.")

    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    matched_ids: List[int] = []

    for i, mid in enumerate(ids.flatten()):
        mid_int = int(mid)
        if mid_int in template_layout:
            c = np.mean(corners[i][0], axis=0)  # tâm hiện tại
            src_pts.append(c.astype(np.float32))
            dst_pts.append(
                np.array(template_layout[mid_int], dtype=np.float32)
            )  # tâm chuẩn
            matched_ids.append(mid_int)

    log.debug(f"[Warp] matched_ids={matched_ids}")
    if len(src_pts) < 4:
        raise RuntimeError("Matched markers < 4; cannot compute homography.")

    src = np.stack(src_pts, axis=0)
    dst = np.stack(dst_pts, axis=0)
    log.debug(f"[Warp] src_pts={src.tolist()}")
    log.debug(f"[Warp] dst_pts={dst.tolist()}")

    H, mask = cv.findHomography(src, dst, cv.RANSAC, ransacReprojThreshold=3.0)
    log.debug(
        f"[Warp] H={None if H is None else H.tolist()}, "
        f"mask={None if mask is None else mask.ravel().tolist()}"
    )
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    warped = cv.warpPerspective(img, H, out_size)
    log.debug(f"[Warp] warped_shape={warped.shape}")
    return warped


def auto_crop_using_markers(
    warped: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Cắt gọn vùng bài làm dựa trên tứ giác bao quanh các marker biên.

    Tôi dùng bounding box + padding vì đơn giản mà vẫn tin cậy khi marker đặt quanh biên.
    """
    log.debug(f"[Crop] warped_shape={warped.shape}")
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(
        f"[Crop] detected_ids={None if ids is None else ids.flatten().tolist()}"
    )

    if ids is None or len(ids) < 4:
        log.warning("[Crop] Not enough markers; returning full image.")
        H, W = warped.shape[:2]
        return warped, warped.copy(), (0, 0, W, H)

    centers = np.array([np.mean(c[0], axis=0) for c in corners])
    log.debug(f"[Crop] centers={centers.tolist()}")

    # Phân loại góc dựa theo tổng toạ độ và hiệu để lấy tl,tr,bl,br
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1).ravel()
    tl, br = centers[np.argmin(s)], centers[np.argmax(s)]
    tr, bl = centers[np.argmin(diff)], centers[np.argmax(diff)]
    log.debug(
        f"[Crop] tl={tl.tolist()}, tr={tr.tolist()}, "
        f"bl={bl.tolist()}, br={br.tolist()}"
    )

    x_min, x_max = int(min(tl[0], bl[0])), int(max(tr[0], br[0]))
    y_min, y_max = int(min(tl[1], tr[1])), int(max(bl[1], br[1]))
    pad_x = int(0.03 * (x_max - x_min))
    pad_y = int(0.03 * (y_max - y_min))

    H, W = warped.shape[:2]
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(W, x_max + pad_x)
    y_max = min(H, y_max + pad_y)
    log.debug(
        f"[Crop] bbox=(x_min={x_min}, y_min={y_min}, "
        f"x_max={x_max}, y_max={y_max}), pad=({pad_x},{pad_y})"
    )

    overlay = warped.copy()
    cv.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
    cropped = warped[y_min:y_max, x_min:x_max]
    log.debug(f"[Crop] cropped_shape={cropped.shape}")
    return cropped, overlay, (x_min, y_min, x_max, y_max)


if __name__ == "__main__":
    # Demo markers: đọc ảnh, warp về A4, crop, lưu kết quả.
    # Tôi dùng hard path để bạn chỉ cần chỉnh trong code một lần.
    input_path = "samples/input.jpg"
    layout_path = "samples/template_marker_layout.json"
    out_dir = "debug_markers"

    safe_mkdir(out_dir)

    img = cv.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Nếu chưa có layout, báo lỗi rõ ràng để bạn tự tạo, vì tôi không thể đoán template.
    if not os.path.exists(layout_path):
        raise FileNotFoundError(
            f"Template layout JSON not found: {layout_path}. "
            f"Bạn hãy tạo nó trước bằng extract_template_marker_layout() hoặc pipeline."
        )

    template_layout = load_template_marker_layout(layout_path)

    warped = warp_to_a4(img, template_layout, A4_PX)
    cv.imwrite(f"{out_dir}/warped_A4.png", warped)
    log.info(f"[markers main] Saved {out_dir}/warped_A4.png")

    cropped, overlay, bbox = auto_crop_using_markers(warped)
    cv.imwrite(f"{out_dir}/crop_box.png", overlay)
    cv.imwrite(f"{out_dir}/cropped.png", cropped)
    log.info(f"[markers main] Saved cropped & overlay to {out_dir}, bbox={bbox}")
