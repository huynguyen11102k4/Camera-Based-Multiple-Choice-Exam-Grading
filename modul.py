# path: markers.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set

import cv2 as cv
import numpy as np


APRILTAG_DICT: int = cv.aruco.DICT_APRILTAG_16h5

# (W, H) output A4 ở đơn vị pixel
A4_PX: Tuple[int, int] = (2481, 3509)

TEMPLATE_LAYOUT_FILE: str = "template_marker_layout.json"

# Các window 4-point để refine từng vùng (v6)
WINDOWS_4PTS: List[List[int]] = [
    [13, 14, 15, 16],
    [15, 16, 17, 18],
    [14, 16, 19, 20],
    [16, 18, 20, 21],
]

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
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Vẽ bounding & ID của marker lên ảnh, vì cần debug xem detect có đúng không.
    """
    vis = img_bgr.copy()
    for d in detections:
        pts = d.corners.astype(int)
        cv.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
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


def _draw_windows_overlay(
    img_bgr: np.ndarray,
    template_layout: Dict[int, List[float]],
    windows: List[List[int]],
) -> np.ndarray:
    """
    Vẽ overlay các cụm WINDOW_4PTS với màu khác nhau, vì bạn muốn debug từng vùng rõ ràng.
    """
    vis = img_bgr.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (128, 0, 255),
    ]
    for wi, ids in enumerate(windows):
        pts = []
        for mid in ids:
            if mid in template_layout:
                pts.append(template_layout[mid])
        if len(pts) < 3:
            continue
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        color = colors[wi % len(colors)]
        cv.polylines(vis, [pts_arr], isClosed=True, color=color, thickness=2)
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
    restrict_ids: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gom các cặp (src, dst) theo layout, vì đây là input chung cho H & IDW.
    """
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    for d in detections:
        if d.id in template_layout:
            if (restrict_ids is not None) and (d.id not in restrict_ids):
                continue
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
    """
    H global: từ ảnh thô → A4, vì cần một bước cơ bản đưa giấy về đúng frame template.
    """
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
    grid_shape: Tuple[int, int] = (24, 32),
    idw_power: float = 2.0,
    idw_eps: float = 1e-3,
    restrict_ids: Optional[Set[int]] = None,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "step4",
) -> np.ndarray:
    """
    IDW refine toàn ảnh hoặc theo subset marker, vì muốn bù méo phi tuyến dựa trên residual.
    """
    H_img, W_img = warped.shape[:2]
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)
    log.info(f"[RefineIDW] detected_ids={[d.id for d in detections]}")

    # Gom src/dst chỉ theo các marker có trong layout (và restrict_ids nếu có)
    src_pts_list: List[np.ndarray] = []
    dst_pts_list: List[np.ndarray] = []

    for d in detections:
        if d.id in template_layout:
            if restrict_ids is not None and d.id not in restrict_ids:
                continue
            src_pts_list.append(d.center.astype(np.float32))
            dst_pts_list.append(np.asarray(template_layout[d.id], np.float32))

    if len(src_pts_list) < 4:
        log.warning(
            f"[RefineIDW] matched markers < 4 (got {len(src_pts_list)}), skip refine."
        )
        return warped

    src_pts = np.stack(src_pts_list, axis=0)  # (N,2)
    dst_pts = np.stack(dst_pts_list, axis=0)  # (N,2)

    residuals = src_pts - dst_pts             # vector cần bù (N,2)
    avg_res = float(np.linalg.norm(residuals, axis=1).mean())
    log.info(
        f"[RefineIDW] N={len(residuals)}, avg_residual={avg_res:.2f}px, restrict_ids={restrict_ids}"
    )

    gx, gy = grid_shape
    map_dx_coarse = np.zeros((gy + 1, gx + 1), dtype=np.float32)
    map_dy_coarse = np.zeros((gy + 1, gx + 1), dtype=np.float32)

    dst_pts_f = dst_pts.astype(np.float32)

    # Tính IDW trên lưới coarse, vì tính trực tiếp từng pixel sẽ rất chậm.
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
            os.path.join(debug_dir, f"markers_{debug_prefix}_warped_refined.png"),
            refined,
            f"{debug_prefix} warped refined by IDW",
        )

    return refined


def _detect_on_img_bgr(img_bgr: np.ndarray) -> Dict[int, TagDetection]:
    """
    Detect marker trên ảnh BGR, vì refine theo vùng cần biết lại vị trí marker sau global warp.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    dets = detect_tags(gray)
    return {d.id: d for d in dets}


def _compute_bbox_from_template(
    ids: List[int],
    template_layout: Dict[int, List[float]],
    img_shape: Tuple[int, int, int],
    margin: int = 10,
) -> Tuple[int, int, int, int]:
    """
    Tính bounding box trong hệ A4 dựa trên vị trí marker trong template,
    vì warped đã đang gần ~ template nên dùng bbox này dán patch sẽ ổn định.
    """
    H_img, W_img = img_shape[:2]
    xs: List[float] = []
    ys: List[float] = []
    for mid in ids:
        if mid in template_layout:
            x, y = template_layout[mid]
            xs.append(x)
            ys.append(y)
    if not xs:
        raise RuntimeError("No valid ids for bbox, vì không có id nào nằm trong template_layout.")

    x_min = int(np.floor(min(xs))) - margin
    x_max = int(np.ceil(max(xs))) + margin
    y_min = int(np.floor(min(ys))) - margin
    y_max = int(np.ceil(max(ys))) + margin

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W_img, x_max)
    y_max = min(H_img, y_max)
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Invalid bbox computed, vì bbox bị lộn chiều hoặc kích thước âm.")
    return x_min, y_min, x_max, y_max

def refine_idw_patch(
    patch: np.ndarray,
    src_local: np.ndarray,      # (N,2) toạ độ marker sau H-local, trong hệ patch
    dst_local: np.ndarray,      # (N,2) toạ độ marker template trong hệ patch
    grid_shape: Tuple[int, int] = (12, 12),
    idw_power: float = 2.0,
    idw_eps: float = 1e-3,
) -> np.ndarray:
    """
    IDW nội suy trong patch độc lập, chỉ dùng 4 điểm marker của window.

    Các bước:
      - residual = src_local - dst_local
      - tạo lưới coarse
      - nội suy dx, dy theo IDW
      - upscale lên full patch
      - remap patch theo map_x, map_y
    """

    ph, pw = patch.shape[:2]
    N = src_local.shape[0]

    if N < 1:
        return patch

    residuals = (src_local - dst_local).astype(np.float32)  # (N,2)

    gx, gy = grid_shape
    map_dx_coarse = np.zeros((gy+1, gx+1), dtype=np.float32)
    map_dy_coarse = np.zeros((gy+1, gx+1), dtype=np.float32)

    dst = dst_local.astype(np.float32)

    # --- IDW coarse grid ---
    for iy in range(gy+1):
        y = (iy / gy) * (ph - 1)
        for ix in range(gx+1):
            x = (ix / gx) * (pw - 1)

            diff = dst - np.array([x, y], np.float32)
            dist = np.linalg.norm(diff, axis=1) + idw_eps
            w = 1.0 / (dist ** idw_power)

            w_sum = w.sum()
            if w_sum < 1e-6:
                dx = dy = 0.0
            else:
                wn = (w[:, None] / w_sum)  # normalize
                delta = (wn * residuals).sum(axis=0)
                dx, dy = float(delta[0]), float(delta[1])

            map_dx_coarse[iy, ix] = dx
            map_dy_coarse[iy, ix] = dy

    # --- upscale lên entire patch ---
    map_dx = cv.resize(map_dx_coarse, (pw, ph), interpolation=cv.INTER_LINEAR)
    map_dy = cv.resize(map_dy_coarse, (pw, ph), interpolation=cv.INTER_LINEAR)

    # --- tạo map_x, map_y ---
    xs, ys = np.meshgrid(
        np.arange(pw, dtype=np.float32),
        np.arange(ph, dtype=np.float32)
    )

    map_x = np.clip(xs + map_dx, 0, pw - 1).astype(np.float32)
    map_y = np.clip(ys + map_dy, 0, ph - 1).astype(np.float32)

    # --- remap patch ---
    refined = cv.remap(
        patch,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    return refined


def refine_warp_multi_region(
    warped: np.ndarray,
    template_layout: Dict[int, List[float]],
    windows_4pts: List[List[int]] = WINDOWS_4PTS,
    use_local_idw: bool = False,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """
    V6 (sửa): Multi-H thật sự local theo từng window 4 điểm.

    Ý tưởng:
      - Làm việc trên bản copy 'base' của warped, vì không muốn thay trực tiếp khi còn đang tính toán.
      - Với mỗi window [13,14,15,16], ...:
          + Lấy bbox trong hệ template → bbox trong ảnh warped, vì warped đã gần template rồi.
          + Cắt patch = base[y_min:y_max, x_min:x_max], vì chỉ muốn warp vùng này.
          + Chuyển center marker từ toạ độ global → local patch, vì homography local cần toạ độ patch.
          + Tính H_local từ src_local -> dst_local, vì vậy 4 marker sẽ được kéo về đúng vị trí template (trong patch).
          + warpPerspective chỉ trên patch với H_local, vì như vậy các vùng ngoài patch không bị ảnh hưởng.
          + Dán patch đã warp ngược trở lại vị trí cũ trong base, vì đây là ảnh kết quả chính.
      - (Tuỳ chọn) Local IDW trên patch sau H, nhưng mặc định tắt vì H đã khớp 4 điểm rồi.
    """

    base = warped.copy()  # vì không muốn sửa trực tiếp warped gốc
    det_by_id = _detect_on_img_bgr(base)
    log.info(f"[RegionRefine] detected_ids={sorted(det_by_id.keys())}")

    if debug_dir is not None:
        safe_mkdir(debug_dir)

    H_img, W_img = base.shape[:2]

    for wi, ids in enumerate(windows_4pts):
        # Lọc ra các id thực sự có mặt cả trong detection lẫn template
        ids_set = [mid for mid in ids if (mid in det_by_id and mid in template_layout)]
        if len(ids_set) < 4:
            log.warning(
                f"[RegionRefine] window {ids} có {len(ids_set)} id hợp lệ (<4), bỏ qua vì không đủ để tính H local."
            )
            continue

        # 1. Tính bbox trong hệ template (≈ hệ warped) để cắt patch
        try:
            x_min, y_min, x_max, y_max = _compute_bbox_from_template(
                ids_set, template_layout, base.shape
            )
        except RuntimeError as e:
            log.warning(
                f"[RegionRefine] tính bbox fail cho window {ids_set}: {e}, bỏ qua vì bbox không hợp lệ."
            )
            continue

        patch = base[y_min:y_max, x_min:x_max]
        patch_h, patch_w = patch.shape[:2]
        if patch_h <= 0 or patch_w <= 0:
            log.warning(
                f"[RegionRefine] patch rỗng cho bbox ({x_min},{y_min})-({x_max},{y_max}), bỏ qua vì không có gì để warp."
            )
            continue

        # 2. Chuẩn bị toạ độ local (patch) cho src/dst, vì H local hoạt động trong hệ patch
        offset = np.array([x_min, y_min], dtype=np.float32)

        src_local = np.stack(
            [det_by_id[mid].center - offset for mid in ids_set], axis=0
        ).astype(np.float32)
        dst_local = np.stack(
            [np.asarray(template_layout[mid], np.float32) - offset for mid in ids_set],
            axis=0,
        )

        # 3. Tính H local trong toạ độ patch, vì ta muốn một H hoạt động thuần tuý trên patch
        H_loc, mask = cv.findHomography(src_local, dst_local, cv.RANSAC, 3.0)
        if H_loc is None or mask is None or int(mask.sum()) < 4:
            log.warning(
                f"[RegionRefine] local H failed cho window {ids_set}, bỏ qua vì không đủ inlier."
            )
            continue

        log.info(
            f"[RegionRefine] window={ids_set}, inliers={int(mask.sum())}/{len(mask)}, "
            f"bbox=({x_min},{y_min})-({x_max},{y_max})"
        )

        # 4. Warp CHỈ patch bằng H_local, vì đây là bước đảm bảo local-H không nhiễu vùng khác
        patch_corrected = cv.warpPerspective(
            patch,
            H_loc,
            (patch_w, patch_h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REPLICATE,
        )

        # 5. (Optional) Local IDW trong patch – mặc định mình tắt, vì với đúng 4 điểm thì H đã khớp hoàn toàn
        #    Nếu bạn muốn bật IDW local, có thể thêm 1 hàm refine_warp_idw_patch ở đây.
        if use_local_idw:
            patch_corrected = refine_idw_patch(
                patch_corrected,
                src_local,  # toạ độ sau H-local
                dst_local,  # toạ độ chuẩn template, cùng hệ patch
                grid_shape=(12, 12),
                idw_power=4.0
            )

        # 6. Dán patch đã warp trở lại ảnh base, vì đây là kết quả final cho vùng này
        base[y_min:y_max, x_min:x_max] = patch_corrected

        if debug_dir is not None:
            _safe_imwrite(
                os.path.join(
                    debug_dir, f"markers_region_{wi+1:02d}_patched_localH.png"
                ),
                base,
                f"region {wi+1} patched by local-H",
            )

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "markers_region_final_localH.png"),
            base,
            "final after multi-region local-H refine",
        )

    return base


def warp_to_a4(
    img: np.ndarray,
    template_layout: Dict[int, List[float]],
    out_size: Tuple[int, int] = A4_PX,
    use_global_refine: bool = True,
    use_region_refine: bool = True,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    warped = warp_to_a4_single_h(img, template_layout, out_size, debug_dir=debug_dir)

    if use_global_refine:
        warped = refine_warp_idw(warped, template_layout, debug_dir=debug_dir)

    if use_region_refine:
        warped = refine_warp_multi_region(
            warped,
            template_layout,
            windows_4pts=WINDOWS_4PTS,
            use_local_idw=False,   # để local-H độc lập, chưa IDW patch
            debug_dir=debug_dir,
        )

    return warped


if __name__ == "__main__":
    # Demo nhỏ, vì cần có entrypoint để test nhanh.
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

    warped = warp_to_a4(
        img,
        layout,
        out_size=A4_PX,
        use_global_refine=True,
        use_region_refine=True,
        debug_dir=out_dir,
    )

    warped_path = os.path.join(out_dir, "warped_a4.png")
    _safe_imwrite(warped_path, warped, "final warped A4 (WarpEngine v6)")

    log.info(f"[Demo] Done. Check debug images in {out_dir}")
