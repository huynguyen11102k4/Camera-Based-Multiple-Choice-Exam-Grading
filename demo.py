# path: omr_pipeline_roi_any.py
from __future__ import annotations

from numpy import ndarray

"""
OMR pipeline hỗ trợ ROI hình chữ nhật và hình tròn:
- Đọc layout mốc ArUco từ template, warp về khuôn A4, auto-crop vùng bài làm.
- Nạp ROI (tự nhận dạng định dạng JSON hình tròn/rect).
- Nhị phân hoá ảnh, đo tỷ lệ tô trong ROI để suy ra đáp án.
- Vẽ kết quả chấm và lưu các bước debug.

Lưu ý:
- `CIRCLE_STROKE` điều khiển độ dày nét vẽ vòng tròn khi hiển thị kết quả.
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np

# === Constants ===
A4_PX = (2480, 3508)  # 300 DPI portrait (W, H)
ARUCO_DICT = cv.aruco.DICT_4X4_50
TEMPLATE_LAYOUT_FILE = "template_marker_layout.json"
ROI_CALIB_FILE = "circle_rois.json"
DEBUG_DIR = "debug_steps"

CIRCLE_STROKE = 4  # độ dày vòng tròn khi vẽ kết quả

# === Types ===
@dataclass
class ROI:
    """Vùng hình chữ nhật cho 1 lựa chọn.

    Attributes:
        y_top: Biên trên (px).
        y_bottom: Biên dưới (px).
        x_left: Biên trái (px).
        x_right: Biên phải (px).
        question: Chỉ số câu (1-based).
        option: Chỉ số đáp án (0-based).
    """
    y_top: int
    y_bottom: int
    x_left: int
    x_right: int
    question: int
    option: int

    def clamp_to(self, width: int, height: int) -> "ROI":
        """Kẹp ROI vào trong kích thước ảnh và sửa thứ tự nếu bị đảo."""
        xL = max(0, min(self.x_left, width - 1))
        xR = max(0, min(self.x_right, width))
        yT = max(0, min(self.y_top, height - 1))
        yB = max(0, min(self.y_bottom, height))
        if xR < xL:
            xL, xR = xR, xL  # tránh rect âm
        if yB < yT:
            yT, yB = yB, yT
        return ROI(yT, yB, xL, xR, self.question, self.option)

    def to_tuple(self) -> Tuple[int, int, int, int, int, int]:
        """Trả về tuple (yT,yB,xL,xR,q,o)."""
        return self.y_top, self.y_bottom, self.x_left, self.x_right, self.question, self.option

    @staticmethod
    def from_seq(seq: Sequence[int]) -> "ROI":
        """Tạo ROI từ chuỗi (yT,yB,xL,xR,q,o)."""
        yT, yB, xL, xR, q, o = seq
        return ROI(int(yT), int(yB), int(xL), int(xR), int(q), int(o))


@dataclass
class CircleROI:
    """Vùng hình tròn cho 1 lựa chọn."""
    cx: int
    cy: int
    r: int
    question: int
    option: int

    def to_rect_tuple(self) -> Tuple[int, int, int, int, int, int]:
        """Chuyển tròn sang tuple rect tương thích (yT,yB,xL,xR,q,o)."""
        xL = self.cx - self.r
        xR = self.cx + self.r
        yT = self.cy - self.r
        yB = self.cy + self.r
        return yT, yB, xL, xR, self.question, self.option


AnyROI = Union[ROI, CircleROI]

# === Logging ===
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("omr")

# === File utils ===
def safe_mkdir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)

def safe_cleanup_dir(path: str) -> None:
    """Xoá file trong thư mục (không xoá thư mục)."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass

# === Display helper (zoom/pan) ===
def show_and_save_step(idx: int, name: str, img: np.ndarray, folder: str) -> None:
    """Lưu ảnh từng bước và mở cửa sổ xem có zoom/pan để kiểm tra."""
    safe_mkdir(folder)
    path = os.path.join(folder, f"{idx:02d}_{name}.png")
    ok = cv.imwrite(path, img)
    log.debug(f"[Step {idx}] Saved step '{name}' -> {path}, ok={ok}, shape={img.shape}")

    window = f"{idx:02d}_{name}"
    h, w = img.shape[:2]
    screen_w, screen_h = 1920, 1080
    scale = min(screen_w / w, screen_h / h, 1.0)
    base_display = cv.resize(img, (int(w * scale), int(h * scale)))
    win_w = int(min(screen_w, base_display.shape[1]) * 0.9)
    win_h = int(min(screen_h, base_display.shape[0]) * 0.9)
    log.debug(f"[Step {idx}] window='{window}', base_display={base_display.shape}, win=({win_w},{win_h}), scale={scale:.3f}")

    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.moveWindow(window, 800, 0)
    cv.resizeWindow(window, win_w, win_h)

    # Trạng thái xem
    zoom = 0.9
    pan_x = 0.0
    pan_y = 0.0
    dragging = False
    start_x = start_y = 0

    def update():
        """Cập nhật vùng hiển thị theo zoom/pan, tự pad nếu thiếu."""
        nonlocal zoom, pan_x, pan_y
        zoom = float(np.clip(zoom, 0.1, 10.0))
        view_w, view_h = win_w, win_h
        zoomed = cv.resize(base_display, None, fx=zoom, fy=zoom)
        zh, zw = zoomed.shape[:2]
        pan_x = float(np.clip(pan_x, 0, max(0, zw - view_w)))
        pan_y = float(np.clip(pan_y, 0, max(0, zh - view_h)))
        x0, y0 = int(pan_x), int(pan_y)
        x1, y1 = min(x0 + view_w, zw), min(y0 + view_h, zh)
        cropped = zoomed[y0:y1, x0:x1]
        pad_h = max(0, view_h - cropped.shape[0])
        pad_w = max(0, view_w - cropped.shape[1])
        if pad_h or pad_w:
            cropped = cv.copyMakeBorder(cropped, 0, pad_h, 0, pad_w, cv.BORDER_CONSTANT, value=(0, 0, 0))
        cv.imshow(window, cropped)

    def on_mouse(event, x, y, flags, _param):
        """Chuột trái kéo để pan, lăn để zoom."""
        nonlocal dragging, start_x, start_y, pan_x, pan_y, zoom
        if event == cv.EVENT_LBUTTONDOWN:
            dragging = True
            start_x, start_y = x, y
        elif event == cv.EVENT_MOUSEMOVE and dragging:
            dx, dy = x - start_x, y - start_y
            pan_x -= dx
            pan_y -= dy
            start_x, start_y = x, y
            update()
        elif event == cv.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv.EVENT_MOUSEWHEEL:
            zoom *= 1.1 if flags > 0 else 1 / 1.1
            update()

    cv.setMouseCallback(window, on_mouse)
    log.info(f"[Step {idx}] {name} — Enter=next, Esc=quit")
    update()
    while True:
        key = cv.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # Esc
            cv.destroyAllWindows()
            raise SystemExit("Stopped by user.")
    cv.destroyAllWindows()

# === ArUco helpers ===
def build_aruco_detector() -> cv.aruco.ArucoDetector:
    """Khởi tạo detector với tinh chỉnh subpixel để ổn định góc."""
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    return cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), params)

def detect_markers(gray: np.ndarray) -> tuple[Sequence[ndarray], ndarray]:
    """Phát hiện marker ArUco trên ảnh xám."""
    detector = build_aruco_detector()
    corners, ids, _rejected = detector.detectMarkers(gray)
    return corners, ids

# === Template layout I/O ===
def extract_template_marker_layout(img_path: str, save_path: str = TEMPLATE_LAYOUT_FILE) -> None:
    """Tạo layout mốc từ ảnh template, lưu {marker_id: [cx, cy]}."""
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {img_path}")
    log.debug(f"[Template] Input: {img_path}, shape={None if img is None else img.shape}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(f"[Template] Detected ids: {None if ids is None else ids.flatten().tolist()}")
    if ids is None or len(ids) < 4:
        raise RuntimeError("Template must contain ≥4 ArUco markers.")
    layout: Dict[int, List[float]] = {}
    for i, mid in enumerate(ids.flatten()):
        center = np.mean(corners[i][0], axis=0).tolist()  # trung bình 4 đỉnh
        log.debug(f"[Template] id={int(mid)} center={center}")
        layout[int(mid)] = center
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    log.info(f"[Template] Saved {len(layout)} markers -> {save_path}")

def load_template_marker_layout(path: str = TEMPLATE_LAYOUT_FILE) -> Dict[int, List[float]]:
    """Đọc layout mốc {id: [cx, cy]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): [float(x) for x in v] for k, v in data.items()}

# === Warp & crop ===
def warp_to_a4(img: np.ndarray, template_layout: Dict[int, List[float]], out_size: Tuple[int, int] = A4_PX) -> np.ndarray:
    """Ước lượng homography từ vị trí marker hiện tại sang layout chuẩn và warp về A4."""
    log.debug(f"[Warp] input_shape={img.shape}, out_size={out_size}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(f"[Warp] detected_ids={None if ids is None else ids.flatten().tolist()}")
    if ids is None or len(ids) < 4:
        raise RuntimeError("Not enough markers to warp to A4.")
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    matched_ids: List[int] = []
    for i, mid in enumerate(ids.flatten()):
        if int(mid) in template_layout:
            c = np.mean(corners[i][0], axis=0)
            src_pts.append(c)                       # tâm hiện tại
            dst_pts.append(np.array(template_layout[int(mid)], dtype=np.float32))  # tâm chuẩn
            matched_ids.append(int(mid))
    log.debug(f"[Warp] matched_ids={matched_ids}")
    if len(src_pts) < 4:
        raise RuntimeError("Matched markers < 4; cannot compute homography.")
    src = np.array(src_pts, np.float32)
    dst = np.array(dst_pts, np.float32)
    log.debug(f"[Warp] src_pts={src.tolist()}")
    log.debug(f"[Warp] dst_pts={dst.tolist()}")
    H, mask = cv.findHomography(src, dst, cv.RANSAC, ransacReprojThreshold=3.0)
    log.debug(f"[Warp] H={None if H is None else H.tolist()}, mask={None if mask is None else mask.ravel().tolist()}")
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    warped = cv.warpPerspective(img, H, out_size)
    log.debug(f"[Warp] warped_shape={warped.shape}")
    return warped

def auto_crop_using_markers(warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Cắt gọn vùng bài làm dựa trên tứ giác bao quanh các marker biên."""
    log.debug(f"[Crop] warped_shape={warped.shape}")
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    log.debug(f"[Crop] detected_ids={None if ids is None else ids.flatten().tolist()}")
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
    log.debug(f"[Crop] tl={tl.tolist()}, tr={tr.tolist()}, bl={bl.tolist()}, br={br.tolist()}")

    x_min, x_max = int(min(tl[0], bl[0])), int(max(tr[0], br[0]))
    y_min, y_max = int(min(tl[1], tr[1])), int(max(bl[1], br[1]))
    pad_x = int(0.03 * (x_max - x_min))
    pad_y = int(0.03 * (y_max - y_min))

    H, W = warped.shape[:2]
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(W, x_max + pad_x)
    y_max = min(H, y_max + pad_y)
    log.debug(f"[Crop] bbox=(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}), pad=({pad_x},{pad_y})")

    overlay = warped.copy()
    cv.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
    cropped = warped[y_min:y_max, x_min:x_max]
    log.debug(f"[Crop] cropped_shape={cropped.shape}")
    return cropped, overlay, (x_min, y_min, x_max, y_max)

# === ROI loaders (auto-detect circle/rect) ===
def load_any_rois(path: str, img_w: int, img_h: int) -> Tuple[List[AnyROI], str]:
    """Đọc JSON ROI, tự nhận dạng định dạng.

    Trả về:
        (rois, kind) với kind {"circle","rect"}

    Hỗ trợ:
      - [{"cx":..,"cy":..,"r":..,"question":..,"option":..}, ...]
      - [[yT,yB,xL,xR,q,o], ...] hoặc [{"y_top":..,"y_bottom":.., ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.debug(f"[ROI] load path={path}, count={0 if not data else len(data)}, img_size=({img_w},{img_h})")
    if not data:
        return [], "unknown"

    first = data[0]
    rois: List[AnyROI] = []

    if isinstance(first, dict) and {"cx", "cy", "r", "question", "option"} <= set(first.keys()):
        # circle
        log.debug("[ROI] format=circle (dict)")
        for idx, d in enumerate(data):
            r = CircleROI(int(d["cx"]), int(d["cy"]), int(d["r"]), int(d["question"]), int(d["option"]))
            orig_r = r.r
            # Kẹp bán kính để không vượt biên ảnh
            r.r = int(max(1, min(r.r, r.cx, img_w - r.cx, r.cy, img_h - r.cy)))
            if r.r != orig_r:
                log.debug(f"[ROI] circle[{idx}] clamped radius from {orig_r} to {r.r} for cx={r.cx}, cy={r.cy}")
            log.debug(f"[ROI] circle[{idx}] (q={r.question}, opt={r.option}) center=({r.cx},{r.cy}), r={r.r}")
            rois.append(r)
        log.info(f"[ROI] Loaded {len(rois)} circle ROIs")
        return rois, "circle"

    elif isinstance(first, (list, tuple)) and len(first) == 6:
        # rect (list)
        log.debug("[ROI] format=rect (list)")
        for idx, seq in enumerate(data):
            roi = ROI.from_seq(seq).clamp_to(img_w, img_h)
            log.debug(f"[ROI] rect[{idx}] (q={roi.question}, opt={roi.option}) y=({roi.y_top},{roi.y_bottom}), x=({roi.x_left},{roi.x_right})")
            rois.append(roi)
        log.info(f"[ROI] Loaded {len(rois)} rect ROIs from list format")
        return rois, "rect"

    elif isinstance(first, dict) and {"y_top", "y_bottom", "x_left", "x_right", "question", "option"} <= set(first.keys()):
        # rect (dict)
        log.debug("[ROI] format=rect (dict)")
        for idx, d in enumerate(data):
            roi = ROI(
                int(d["y_top"]), int(d["y_bottom"]), int(d["x_left"]), int(d["x_right"]), int(d["question"]), int(d["option"])
            ).clamp_to(img_w, img_h)
            log.debug(f"[ROI] rect[{idx}] (q={roi.question}, opt={roi.option}) y=({roi.y_top},{roi.y_bottom}), x=({roi.x_left},{roi.x_right})")
            rois.append(roi)
        log.info(f"[ROI] Loaded {len(rois)} rect ROIs from dict format")
        return rois, "rect"

    else:
        raise ValueError("Unsupported ROI JSON format.")

# === Scoring (generic) ===
def detect_marked_generic(thresh: np.ndarray, rois: List[AnyROI], kind: str, fill_threshold: float = 0.35) -> Dict[int, Optional[int]]:
    """Tính tỷ lệ pixel > 0 trong từng ROI và chọn phương án có tỷ lệ cao vượt ngưỡng.

    Args:
        thresh: Ảnh nhị phân (uint8) đã tiền xử lý.
        rois: Danh sách ROI tròn hoặc chữ nhật.
        kind: "circle" hoặc "rect".
        fill_threshold: Ngưỡng chọn (0..1).

    Returns:
        {question_index: option_index hoặc None}
    """
    H, W = thresh.shape
    log.debug(f"[Detect] kind={kind}, image_size=({W},{H}), rois={len(rois)}, thr={fill_threshold}")
    results: Dict[int, Optional[int]] = {}
    per_q: Dict[int, List[Tuple[int, float]]] = {}

    if kind == "circle":
        for idx, r in enumerate(rois):  # type: ignore
            r: CircleROI
            # Cắt roi tối thiểu quanh vòng tròn
            x0, x1 = max(0, r.cx - r.r), min(W, r.cx + r.r)
            y0, y1 = max(0, r.cy - r.r), min(H, r.cy + r.r)
            if x1 <= x0 or y1 <= y0:
                ratio = 0.0
                log.debug(f"[Detect] circle[{idx}] invalid ROI for q={r.question}, opt={r.option}, bbox=({x0},{y0},{x1},{y1}) -> ratio=0")
            else:
                roi = thresh[y0:y1, x0:x1]
                yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
                mask = (xx - (r.r)) ** 2 + (yy - (r.r)) ** 2 <= (r.r ** 2)
                sel = roi[mask]
                ratio = float(np.mean(sel > 0)) if sel.size else 0.0
                log.debug(f"[Detect] circle[{idx}] q={r.question}, opt={r.option}, center=({r.cx},{r.cy}), r={r.r}, bbox=({x0},{y0},{x1},{y1}), fill={ratio:.3f}")
            per_q.setdefault(r.question, []).append((r.option, ratio))
    else:  # rect
        for idx, r in enumerate(rois):  # type: ignore
            r: ROI
            y0, y1 = max(0, r.y_top), min(H, r.y_bottom)
            x0, x1 = max(0, r.x_left), min(W, r.x_right)
            if y1 <= y0 or x1 <= x0:
                ratio = 0.0
                log.debug(f"[Detect] rect[{idx}] invalid ROI for q={r.question}, opt={r.option}, y=({y0},{y1}), x=({x0},{x1}) -> ratio=0")
            else:
                roi = thresh[y0:y1, x0:x1]
                ratio = float(np.mean(roi > 0))
                log.debug(f"[Detect] rect[{idx}] q={r.question}, opt={r.option}, y=({y0},{y1}), x=({x0},{x1}), fill={ratio:.3f}")
            per_q.setdefault(r.question, []).append((r.option, ratio))

    for q, opts in per_q.items():
        if not opts:
            results[q] = None
            log.debug(f"[Detect] q={q} no options")
            continue
        opts.sort(key=lambda x: x[1], reverse=True)
        best_opt, best_fill = opts[0]
        chosen = best_opt if best_fill >= fill_threshold else None
        results[q] = chosen
        log.debug(f"[Detect] q={q} opts={[(o, round(v,3)) for o,v in opts]} -> chosen={chosen}, best_fill={best_fill:.3f}")
    return results

def draw_grading_result_generic(img: np.ndarray, rois: List[AnyROI], detected: Dict[int, Optional[int]], answer_key: Dict[int, int], kind: str) -> np.ndarray:
    """Vẽ kết quả chấm lên ảnh: cam=ROI, đỏ=được tô, xanh=đúng đáp án."""
    out = img.copy()
    log.debug(f"[Draw] kind={kind}, rois={len(rois)}")
    if kind == "circle":
        for idx, r in enumerate(rois):  # type: ignore
            r: CircleROI
            color = (255, 140, 0)
            picked = detected.get(r.question)
            reason = "roi"
            if picked == r.option:
                color = (0, 0, 255)
                reason = "picked"
                if answer_key.get(r.question) == r.option:
                    color = (0, 255, 0)
                    reason = "correct"
            log.debug(f"[Draw] circle[{idx}] q={r.question}, opt={r.option}, picked={picked}, key={answer_key.get(r.question)}, color={color}, reason={reason}")
            cv.circle(out, (r.cx, r.cy), r.r, color, CIRCLE_STROKE)
    else:
        for idx, r in enumerate(rois):  # type: ignore
            r: ROI
            # Vẽ vòng tròn nội tiếp để cách hiển thị đồng nhất
            cx = int((r.x_left + r.x_right) / 2)
            cy = int((r.y_top + r.y_bottom) / 2)
            radius = int(0.45 * min(r.x_right - r.x_left, r.y_bottom - r.y_top))
            color = (255, 140, 0)
            picked = detected.get(r.question)
            reason = "roi"
            if picked == r.option:
                color = (0, 0, 255)
                reason = "picked"
                if answer_key.get(r.question) == r.option:
                    color = (0, 255, 0)
                    reason = "correct"
            log.debug(f"[Draw] rect[{idx}] q={r.question}, opt={r.option}, picked={picked}, key={answer_key.get(r.question)}, center=({cx},{cy}), radius={radius}, color={color}, reason={reason}")
            cv.circle(out, (cx, cy), radius, color, CIRCLE_STROKE)
    return out

# === Threshold ===
def build_threshold(gray: np.ndarray) -> np.ndarray:
    """Làm mượt nhẹ và adaptive-threshold để bù sáng không đều."""
    log.debug(f"[Thresh] gray_shape={gray.shape}, dtype={gray.dtype}")
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 33, 5)
    # Stats
    white = int(np.sum(th > 0))
    total = th.size
    log.debug(f"[Thresh] white={white}/{total} ({white/total:.2%})")
    return th

# === Pipeline ===
def pipeline(
    img_path: str,
    template_layout: Dict[int, List[float]],
    sheet_layout_builder,  # callable: (W, H) -> List[ROI or tuples]
    roi_calib_file: str = ROI_CALIB_FILE,
    debug_dir: str = DEBUG_DIR,
    interactive: bool = True,
) -> None:
    """Pipeline đầy đủ từ ảnh quét đến kết quả chấm có hiển thị/từng bước."""
    log.debug(f"[Pipe] start img_path={img_path}, template_layout_ids={sorted(list(template_layout.keys()))}, debug_dir={debug_dir}, interactive={interactive}")
    safe_mkdir(debug_dir)
    safe_cleanup_dir(debug_dir)

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")
    log.debug(f"[Pipe] original_shape={img.shape}")

    show_and_save_step(1, "original", img, debug_dir)

    # Chuẩn hoá theo A4 qua homography
    warped = warp_to_a4(img, template_layout, A4_PX)
    show_and_save_step(2, "warped_A4", warped, debug_dir)

    # Cắt gọn vùng bài
    cropped, overlay, bbox = auto_crop_using_markers(warped)
    log.debug(f"[Pipe] crop_bbox={bbox}")
    show_and_save_step(3, "crop_box", overlay, debug_dir)
    show_and_save_step(4, "cropped", cropped, debug_dir)

    # Nhị phân hoá để đo phần tô
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    log.debug(f"[Pipe] cropped_shape={cropped.shape}, gray_shape={gray.shape}")
    thresh = build_threshold(gray)
    show_and_save_step(5, "threshold", thresh, debug_dir)

    H, W = cropped.shape[:2]
    log.debug(f"[Pipe] working_size=({W},{H})")

    rois_any: List[AnyROI] = []
    kind: str = "rect"

    if os.path.exists(roi_calib_file):
        rois_any, kind = load_any_rois(roi_calib_file, W, H)
        log.info(f"Loaded ROI calibration ({kind}): {roi_calib_file}")
    else:
        # Fallback: dùng layout builder cũ (rect)
        raw_rois = sheet_layout_builder(W, H)
        log.debug(f"[Pipe] built {len(raw_rois)} raw ROIs from layout builder")
        rois_any = [ROI.from_seq(r) if not isinstance(r, ROI) else r for r in raw_rois]
        rois_any = [r.clamp_to(W, H) for r in rois_any]  # type: ignore
        if interactive:
            adjusted = interactive_roi_adjustment(cropped, rois_any, save_path=roi_calib_file)  # type: ignore
            if adjusted is not None:
                rois_any = adjusted  # type: ignore
        kind = "rect"
    log.debug(f"[Pipe] ROI kind={kind}, count={len(rois_any)}")

    # Suy ra đáp án được tô
    detected = detect_marked_generic(thresh, rois_any, kind, fill_threshold=0.35)
    log.debug(f"[Pipe] detected={detected}")

    # Sinh đáp án mẫu ngẫu nhiên để demo
    random.seed(42)
    q_ids = {(r.question if isinstance(r, CircleROI) else r.question) for r in rois_any}
    n_q = max(q_ids) if q_ids else 50
    n_opts = max((r.option for r in rois_any), default=4) + 1 if rois_any else 5
    answer_key = {i + 1: random.randint(0, n_opts - 1) for i in range(n_q)}
    log.debug(f"[Pipe] answer_key(sample 10)={dict(list(answer_key.items())[:10])} ... total={len(answer_key)}")

    score = grade_exam(detected, answer_key)
    log.info(f"Score: {score:.2f}/10")

    # Vẽ kết quả cuối
    graded = draw_grading_result_generic(cropped, rois_any, detected, answer_key, kind)
    show_and_save_step(6, "final_marked", graded, debug_dir)
    out_path = os.path.join(debug_dir, "final_marked.png")
    ok = cv.imwrite(out_path, graded)
    log.info(f"Saved: {out_path} (ok={ok})")

# === Legacy helpers reused ===
def grade_exam(detected: Dict[int, Optional[int]], answer_key: Dict[int, int]) -> float:
    """Tính điểm thang 10 theo tỷ lệ đúng/số câu."""
    if not answer_key:
        log.debug("[Grade] empty answer_key -> 0")
        return 0.0
    details = {q: (detected.get(q), ans) for q, ans in answer_key.items()}
    correct = sum(1 for q, (pred, ans) in details.items() if pred == ans)
    wrong = [(q, pred, ans) for q, (pred, ans) in details.items() if pred is not None and pred != ans]
    missing = [q for q, (pred, _ans) in details.items() if pred is None]
    log.debug(f"[Grade] correct={correct}/{len(answer_key)}, wrong={len(wrong)}, missing={len(missing)}")
    if wrong:
        log.debug(f"[Grade] wrong_samples={wrong[:10]}")
    if missing:
        log.debug(f"[Grade] missing_samples={missing[:10]}")
    return correct / len(answer_key) * 10.0

def interactive_roi_adjustment(img: np.ndarray, rois: List[ROI], save_path: str = ROI_CALIB_FILE) -> Optional[List[ROI]]:
    """Chỗ móc để chỉnh ROI (rect) thủ công khi chưa có JSON tròn."""
    return rois

# === Main / CLI ===
def main() -> None:
    """CLI: chạy pipeline với tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="OMR pipeline (rect/circle ROI auto-detect).")
    parser.add_argument("--template", default="template_scan.png", help="Template image path")
    parser.add_argument("--input", default="Demo6.jpg", help="Scanned sheet path")
    parser.add_argument("--template-layout", default=TEMPLATE_LAYOUT_FILE, help="Template layout JSON")
    parser.add_argument("--roi-calib", default=ROI_CALIB_FILE, help="ROI JSON (circle or rect)")
    parser.add_argument("--debug-dir", default=DEBUG_DIR, help="Debug output directory")
    parser.add_argument("--no-ui", action="store_true", help="Disable interactive windows")
    args = parser.parse_args()

    log.debug(f"[CLI] args={vars(args)}")
    from SheetLayout import SheetLayout

    if not os.path.exists(args.template_layout):
        extract_template_marker_layout(args.template, args.template_layout)
    template_layout = load_template_marker_layout(args.template_layout)
    log.debug(f"[CLI] loaded template_layout ids={sorted(list(template_layout.keys()))}")

    layout = SheetLayout()
    sheet_layout_builder = layout.buildRois

    pipeline(
        img_path=args.input,
        template_layout=template_layout,
        sheet_layout_builder=sheet_layout_builder,
        roi_calib_file=args.roi_calib,
        debug_dir=args.debug_dir,
        interactive=not args.no_ui,
    )

if __name__ == "__main__":
    main()
