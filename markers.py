# path: markers.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np


APRILTAG_DICT: int = cv.aruco.DICT_APRILTAG_16h5

A4_PX: Tuple[int, int] = (2481, 3509)

TEMPLATE_LAYOUT_FILE: str = "template_marker_layout.json"


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_imwrite(path: str, img: np.ndarray, note: str = "") -> None:
    ok = cv.imwrite(path, img)
    if ok:
        if note:
            log.info(f"[DebugImg] Saved {note} -> {path}")
        else:
            log.info(f"[DebugImg] Saved -> {path}")
    else:
        log.warning(f"[DebugImg] Failed to save image -> {path}")


@dataclass
class TagDetection:
    id: int                # ID của marker
    corners: np.ndarray    # 4 đỉnh (4, 2)
    center: np.ndarray     # tâm (2,)


@lru_cache(maxsize=1)
def _build_detector() -> cv.aruco.ArucoDetector:
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

    dictionary = cv.aruco.getPredefinedDictionary(APRILTAG_DICT)
    detector = cv.aruco.ArucoDetector(dictionary, params)
    log.info(f"[Detector] Init ArucoDetector with dict={APRILTAG_DICT}")
    return detector


def detect_tags(gray: np.ndarray) -> List[TagDetection]:
    detector = _build_detector()
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return []

    ids_arr = np.asarray(ids, dtype=np.int32).reshape(-1)

    detections: List[TagDetection] = []
    for i, c in enumerate(corners):
        c_arr = np.asarray(c, dtype=np.float32).reshape(-1, 2)  # (4,2)
        center = c_arr.mean(axis=0)
        detections.append(
            TagDetection(
                id=int(ids_arr[i]),
                corners=c_arr,
                center=center,
            )
        )

    detections.sort(key=lambda d: d.id)
    return detections


def _draw_detections(
    img_bgr: np.ndarray,
    detections: List[TagDetection],
    draw_ids: bool = True,
) -> np.ndarray:

    vis = img_bgr.copy()
    for d in detections:
        pts = d.corners.astype(int)
        cv.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        c = tuple(d.center.astype(int))
        cv.circle(vis, c, 5, (0, 0, 255), -1)
        if draw_ids:
            cv.putText(
                vis,
                str(d.id),
                (c[0] + 5, c[1] - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv.LINE_AA,
            )
    return vis


def extract_template_marker_layout(
    img_path: str,
    save_path: str = TEMPLATE_LAYOUT_FILE,
    debug_dir: Optional[str] = None,
) -> Dict[int, List[float]]:

    import json

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {img_path}")
    log.info(f"[Template] Input: {img_path}, shape={img.shape}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)
    log.info(f"[Template] Detected ids: {[d.id for d in detections]}")

    # STEP 1: lưu ảnh template có tag & id, để xem layout marker có đúng ý không
    if debug_dir is not None:
        safe_mkdir(debug_dir)
        vis = _draw_detections(img, detections)
        _safe_imwrite(
            os.path.join(debug_dir, "markers_step1_template_tags.png"),
            vis,
            "step1 template tags",
        )

    if len(detections) < 4:
        raise RuntimeError("Template must contain ≥4 markers.")

    layout: Dict[int, List[float]] = {}
    seen: set[int] = set()

    for d in detections:
        if d.id in seen:
            raise RuntimeError(f"Duplicate marker id {d.id} in template.")
        seen.add(d.id)
        layout[d.id] = d.center.astype(float).tolist()
        log.debug(f"[Template] id={d.id} center={layout[d.id]}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    log.info(f"[Template] Saved {len(layout)} markers -> {save_path}")
    return layout


def load_template_marker_layout(path: str = TEMPLATE_LAYOUT_FILE) -> Dict[int, List[float]]:

    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layout = {int(k): [float(x) for x in v] for k, v in data.items()}
    log.info(f"[Template] loaded ids={sorted(layout.keys())}")
    return layout


def _collect_correspondences(
    detections: List[TagDetection],
    template_layout: Dict[int, List[float]],
) -> Tuple[np.ndarray, np.ndarray]:

    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    for d in detections:
        if d.id in template_layout:
            src.append(d.center.astype(np.float32))
            dst.append(np.asarray(template_layout[d.id], dtype=np.float32))

    if not src:
        raise RuntimeError("No common marker ids between image and layout.")

    src_pts = np.stack(src, axis=0)
    dst_pts = np.stack(dst, axis=0)
    return src_pts, dst_pts


def warp_to_a4_single_h(
    img: np.ndarray,
    template_layout: Dict[int, List[float]],
    out_size: Tuple[int, int] = A4_PX,
    debug_dir: Optional[str] = None,
) -> np.ndarray:

    W_out, H_out = out_size
    if W_out <= 0 or H_out <= 0:
        raise ValueError(f"Invalid out_size={out_size}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)
    log.info(f"[WarpH] detected_ids={[d.id for d in detections]}")

    if debug_dir is not None:
        safe_mkdir(debug_dir)
        vis_input = _draw_detections(img, detections)
        _safe_imwrite(
            os.path.join(debug_dir, "markers_step2_input_tags.png"),
            vis_input,
            "step2 input with tags",
        )

    if len(detections) < 4:
        raise RuntimeError("Need ≥4 markers for homography.")

    src_pts, dst_pts = _collect_correspondences(detections, template_layout)
    if len(src_pts) < 4:
        raise RuntimeError(f"Matched markers < 4 (got {len(src_pts)}).")

    H_sd, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
    if H_sd is None or mask is None or int(mask.sum()) < 4:
        raise RuntimeError("Global homography estimation failed.")

    log.info(f"[WarpH] inliers={int(mask.sum())}/{len(mask)}")

    warped = cv.warpPerspective(img, H_sd, out_size)
    log.info(f"[WarpH] warped_shape={warped.shape}")

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "markers_step3_warped_H_only.png"),
            warped,
            "step3 warped by H only",
        )

    return warped


def refine_warp_idw(
    warped: np.ndarray,
    template_layout: Dict[int, List[float]],
    grid_shape: Tuple[int, int] = (24, 32), # Kích thước lưới nội suy: (row, col) = (10,16), (16,24), (24,32). Số càng lớn thì lưới càng mịn (chi tiết hơn) nhưng tính toán chậm hơn.
    idw_power: float = 2, # Số mũ trong công thức Inverse Distance Weighting (IDW).
        # - power lớn (vd 3, 4): điểm gần ảnh hưởng MẠNH hơn, bề mặt gồ ghề hơn.
        # - power nhỏ (vd 1): điểm xa vẫn còn ảnh hưởng đáng kể, bề mặt mượt hơn.
    idw_eps: float = 1e-3, # Epsilon rất nhỏ để tránh chia cho 0 khi khoảng cách gần như bằng 0.
        # - Nếu có điểm trùng vị trí (d = 0) → chia cho 0 → lỗi.
        # - Thêm eps (1e-3) đảm bảo d không bao giờ là 0 tuyệt đối.

    debug_dir: Optional[str] = None,
) -> np.ndarray:

    H_img, W_img = warped.shape[:2]
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)
    log.info(f"[RefineIDW] detected_ids={[d.id for d in detections]}")

    if len(detections) < 4:
        log.warning("[RefineIDW] Not enough markers, skip refine.")
        return warped

    src_pts_list: List[np.ndarray] = []
    dst_pts_list: List[np.ndarray] = []

    for d in detections:
        if d.id in template_layout:
            src_pts_list.append(d.center.astype(np.float32))
            dst_pts_list.append(np.asarray(template_layout[d.id], np.float32))

    if len(src_pts_list) < 4:
        log.warning("[RefineIDW] matched markers < 4, skip refine.")
        return warped

    src_pts = np.stack(src_pts_list, axis=0)  # (N,2)
    dst_pts = np.stack(dst_pts_list, axis=0)  # (N,2)

    residuals = src_pts - dst_pts             # vector cần bù (N,2)
    avg_res = float(np.linalg.norm(residuals, axis=1).mean())
    log.info(f"[RefineIDW] N={len(residuals)}, avg_residual={avg_res:.2f}px")

    gx, gy = grid_shape
    map_dx_coarse = np.zeros((gy + 1, gx + 1), dtype=np.float32)
    map_dy_coarse = np.zeros((gy + 1, gx + 1), dtype=np.float32)

    dst_pts_f = dst_pts.astype(np.float32)

    for iy in range(gy + 1):
        y = (iy / gy) * (H_img - 1) if gy > 0 else (H_img - 1) / 2.0
        for ix in range(gx + 1):
            x = (ix / gx) * (W_img - 1) if gx > 0 else (W_img - 1) / 2.0

            diff = dst_pts_f - np.array([x, y], dtype=np.float32)
            dists = np.linalg.norm(diff, axis=1) + idw_eps
            weights = 1.0 / (dists ** idw_power)

            w_sum = float(weights.sum())
            if w_sum < 1e-6:
                dx = dy = 0.0
            else:
                w_norm = weights[:, None] / w_sum
                delta = (w_norm * residuals).sum(axis=0)
                dx = float(delta[0])
                dy = float(delta[1])

            map_dx_coarse[iy, ix] = dx
            map_dy_coarse[iy, ix] = dy

    map_dx = cv.resize(map_dx_coarse, (W_img, H_img), interpolation=cv.INTER_LINEAR)
    map_dy = cv.resize(map_dy_coarse, (W_img, H_img), interpolation=cv.INTER_LINEAR)

    xs, ys = np.meshgrid(
        np.arange(W_img, dtype=np.float32),
        np.arange(H_img, dtype=np.float32),
    )
    map_x = xs + map_dx
    map_y = ys + map_dy

    map_x = np.clip(map_x, 0, W_img - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, H_img - 1).astype(np.float32)

    refined = cv.remap(
        warped,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    log.info(f"[RefineIDW] refined_shape={refined.shape}")

    if debug_dir is not None:
        safe_mkdir(debug_dir)
        _safe_imwrite(
            os.path.join(debug_dir, "markers_step4_warped_refined.png"),
            refined,
            "step4 warped refined by IDW",
        )

    return refined


def warp_to_a4(
    img: np.ndarray,
    template_layout: Dict[int, List[float]],
    out_size: Tuple[int, int] = A4_PX,
    use_refine: bool = True,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    warped = warp_to_a4_single_h(img, template_layout, out_size, debug_dir=debug_dir)
    if use_refine:
        warped = refine_warp_idw(warped, template_layout, debug_dir=debug_dir)
    return warped


if __name__ == "__main__":
    template_img_path = "samples/template_scan1.png"
    input_img_path = "samples/photo3.jpg"
    out_dir = "debug_markers"

    safe_mkdir(out_dir)

    if not os.path.exists(TEMPLATE_LAYOUT_FILE):
        log.info("[Demo] Layout JSON not found; extracting from template.")
        extract_template_marker_layout(
            template_img_path,
            TEMPLATE_LAYOUT_FILE,
            debug_dir=out_dir,
        )

    layout = load_template_marker_layout(TEMPLATE_LAYOUT_FILE)

    img = cv.imread(input_img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_img_path}")

    warped = warp_to_a4(img, layout, out_size=A4_PX, use_refine=True, debug_dir=out_dir)

    warped_path = os.path.join(out_dir, "warped_a4.png")
    _safe_imwrite(warped_path, warped, "final warped A4 (for OMR)")

    step5_path = os.path.join(out_dir, "markers_step5_warped_a4.png")
    _safe_imwrite(step5_path, warped, "step5 final warped A4")

    log.info(f"[Demo] Done. Check debug images in {out_dir}")
