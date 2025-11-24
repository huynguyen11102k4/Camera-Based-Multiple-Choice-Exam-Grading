# rois.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np

from core import log


# ========== ROI Types ==========
@dataclass
class ROI:
    """Vùng hình chữ nhật cho 1 lựa chọn."""
    y_top: int
    y_bottom: int
    x_left: int
    x_right: int
    question: int  # 1-based
    option: int    # 0-based

    def clamp_to(self, width: int, height: int) -> "ROI":
        """Kẹp ROI vào trong kích thước ảnh và sửa thứ tự nếu bị đảo.

        Tôi clamp ngay tại đây để các module khác yên tâm sử dụng mà không cần check lại.
        """
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
        """Trả về tuple (yT,yB,xL,xR,q,o) để tương thích layout builder cũ."""
        return (
            self.y_top,
            self.y_bottom,
            self.x_left,
            self.x_right,
            self.question,
            self.option,
        )

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


# ========== ROI Loader (auto-detect circle/rect) ==========
def load_any_rois(
    path: str,
    img_w: int,
    img_h: int,
) -> Tuple[List[AnyROI], str]:
    """Đọc JSON ROI, tự nhận dạng định dạng.

    Trả về:
        (rois, kind) với kind ∈ {"circle", "rect"}.

    Tôi hỗ trợ 3 format để linh hoạt nhập dữ liệu.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.debug(
        f"[ROI] load path={path}, "
        f"count={0 if not data else len(data)}, "
        f"img_size=({img_w},{img_h})"
    )

    if not data:
        return [], "unknown"

    first = data[0]
    rois: List[AnyROI] = []

    # Circle ROI (dict format)
    if isinstance(first, dict) and {
        "cx",
        "cy",
        "r",
        "question",
        "option",
    } <= set(first.keys()):
        log.debug("[ROI] format=circle (dict)")
        for idx, d in enumerate(data):
            r = CircleROI(
                int(d["cx"]),
                int(d["cy"]),
                int(d["r"]),
                int(d["question"]),
                int(d["option"]),
            )
            orig_r = r.r
            # Kẹp bán kính để không vượt biên ảnh
            r.r = int(
                max(
                    1,
                    min(
                        r.r,
                        r.cx,
                        img_w - r.cx,
                        r.cy,
                        img_h - r.cy,
                    ),
                )
            )
            if r.r != orig_r:
                log.debug(
                    f"[ROI] circle[{idx}] clamped radius from "
                    f"{orig_r} to {r.r} for cx={r.cx}, cy={r.cy}"
                )
            log.debug(
                f"[ROI] circle[{idx}] (q={r.question}, opt={r.option}) "
                f"center=({r.cx},{r.cy}), r={r.r}"
            )
            rois.append(r)
        log.info(f"[ROI] Loaded {len(rois)} circle ROIs")
        return rois, "circle"

    # Rect ROI (list format)
    if isinstance(first, (list, tuple)) and len(first) == 6:
        log.debug("[ROI] format=rect (list)")
        for idx, seq in enumerate(data):
            roi = ROI.from_seq(seq).clamp_to(img_w, img_h)
            log.debug(
                f"[ROI] rect[{idx}] (q={roi.question}, opt={roi.option}) "
                f"y=({roi.y_top},{roi.y_bottom}), x=({roi.x_left},{roi.x_right})"
            )
            rois.append(roi)
        log.info(f"[ROI] Loaded {len(rois)} rect ROIs from list format")
        return rois, "rect"

    # Rect ROI (dict format)
    if isinstance(first, dict) and {
        "y_top",
        "y_bottom",
        "x_left",
        "x_right",
        "question",
        "option",
    } <= set(first.keys()):
        log.debug("[ROI] format=rect (dict)")
        for idx, d in enumerate(data):
            roi = ROI(
                int(d["y_top"]),
                int(d["y_bottom"]),
                int(d["x_left"]),
                int(d["x_right"]),
                int(d["question"]),
                int(d["option"]),
            ).clamp_to(img_w, img_h)
            log.debug(
                f"[ROI] rect[{idx}] (q={roi.question}, opt={roi.option}) "
                f"y=({roi.y_top},{roi.y_bottom}), x=({roi.x_left},{roi.x_right})"
            )
            rois.append(roi)
        log.info(f"[ROI] Loaded {len(rois)} rect ROIs from dict format")
        return rois, "rect"

    raise ValueError("Unsupported ROI JSON format.")


# ========== Interactive ROI Adjustment Hook ==========
def interactive_roi_adjustment(
    img: np.ndarray,
    rois: List[ROI],
    save_path: str,
) -> Optional[List[ROI]]:
    """Chỗ móc để chỉnh ROI (rect) thủ công khi chưa có JSON.

    Hiện tại trả về nguyên bản để pipeline vẫn chạy, vì bạn chưa cần GUI chỉnh ROI.
    """
    _ = img, save_path  # tránh warning unused
    return rois


if __name__ == "__main__":
    # Demo rois: đọc input + roi.json, vẽ ROI lên ảnh để kiểm tra.
    from core import ROI_CALIB_FILE, safe_mkdir

    input_path = "samples/input.jpg"
    roi_path = ROI_CALIB_FILE
    out_dir = "debug_rois"

    safe_mkdir(out_dir)

    img = cv.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    H, W = img.shape[:2]

    if not os.path.exists(roi_path):
        raise FileNotFoundError(
            f"ROI JSON not found: {roi_path}. "
            f"Bạn hãy tạo file ROI trước (rect/circle)."
        )

    rois_any, kind = load_any_rois(roi_path, W, H)
    log.info(f"[rois main] Loaded {len(rois_any)} {kind} ROIs")

    # Vẽ ROI: chỉ vẽ viền cam để kiểm tra vị trí, vì không cần grading.
    overlay = img.copy()
    if kind == "circle":
        for r_any in rois_any:
            r: CircleROI = r_any  # type: ignore
            cv.circle(overlay, (r.cx, r.cy), r.r, (0, 165, 255), 2)
    else:
        for r_any in rois_any:
            r: ROI = r_any  # type: ignore
            cv.rectangle(
                overlay,
                (r.x_left, r.y_top),
                (r.x_right, r.y_bottom),
                (0, 165, 255),
                2,
            )

    cv.imwrite(f"{out_dir}/rois_overlay.png", overlay)
    log.info(f"[rois main] Saved {out_dir}/rois_overlay.png")
