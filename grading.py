# grading.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from core import CIRCLE_STROKE, ROI_CALIB_FILE, log, safe_mkdir
from rois import AnyROI, CircleROI, ROI, load_any_rois
from threshold import build_threshold


# ========== Detect Marked ==========
def detect_marked_generic(
    thresh: np.ndarray,
    rois: List[AnyROI],
    kind: str,
    fill_threshold: float = 0.35,
) -> Dict[int, Optional[int]]:
    """Tính tỷ lệ pixel > 0 trong từng ROI và chọn phương án có tỷ lệ cao vượt ngưỡng.

    Tôi dùng mean(roi > 0) vì đây là metric đơn giản, dễ tune và khá robust với noise.
    """
    H, W = thresh.shape
    log.debug(
        f"[Detect] kind={kind}, image_size=({W},{H}), "
        f"rois={len(rois)}, thr={fill_threshold}"
    )
    results: Dict[int, Optional[int]] = {}
    per_q: Dict[int, List[Tuple[int, float]]] = {}

    if kind == "circle":
        for idx, r_any in enumerate(rois):
            r: CircleROI = r_any  # type: ignore
            x0, x1 = max(0, r.cx - r.r), min(W, r.cx + r.r)
            y0, y1 = max(0, r.cy - r.r), min(H, r.cy + r.r)
            if x1 <= x0 or y1 <= y0:
                ratio = 0.0
                log.debug(
                    f"[Detect] circle[{idx}] invalid ROI for q={r.question}, "
                    f"opt={r.option}, bbox=({x0},{y0},{x1},{y1}) -> ratio=0"
                )
            else:
                roi = thresh[y0:y1, x0:x1]
                yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
                # mask vòng tròn trong bbox
                mask = (xx - r.r) ** 2 + (yy - r.r) ** 2 <= (r.r**2)
                sel = roi[mask]
                ratio = float(np.mean(sel > 0)) if sel.size else 0.0
                log.debug(
                    f"[Detect] circle[{idx}] q={r.question}, opt={r.option}, "
                    f"center=({r.cx},{r.cy}), r={r.r}, "
                    f"bbox=({x0},{y0},{x1},{y1}), fill={ratio:.3f}"
                )
            per_q.setdefault(r.question, []).append((r.option, ratio))
    else:  # rect
        for idx, r_any in enumerate(rois):
            r: ROI = r_any  # type: ignore
            y0, y1 = max(0, r.y_top), min(H, r.y_bottom)
            x0, x1 = max(0, r.x_left), min(W, r.x_right)
            if y1 <= y0 or x1 <= x0:
                ratio = 0.0
                log.debug(
                    f"[Detect] rect[{idx}] invalid ROI for q={r.question}, "
                    f"opt={r.option}, y=({y0},{y1}), x=({x0},{x1}) -> ratio=0"
                )
            else:
                roi = thresh[y0:y1, x0:x1]
                ratio = float(np.mean(roi > 0))
                log.debug(
                    f"[Detect] rect[{idx}] q={r.question}, opt={r.option}, "
                    f"y=({y0},{y1}), x=({x0},{x1}), fill={ratio:.3f}"
                )
            per_q.setdefault(r.question, []).append((r.option, ratio))

    # Chọn option có fill cao nhất nếu vượt ngưỡng
    for q, opts in per_q.items():
        if not opts:
            results[q] = None
            log.debug(f"[Detect] q={q} no options")
            continue
        opts.sort(key=lambda x: x[1], reverse=True)
        best_opt, best_fill = opts[0]
        chosen = best_opt if best_fill >= fill_threshold else None
        results[q] = chosen
        log.debug(
            f"[Detect] q={q} opts={[(o, round(v,3)) for o,v in opts]} "
            f"-> chosen={chosen}, best_fill={best_fill:.3f}"
        )
    return results


# ========== Draw Result ==========
def draw_grading_result_generic(
    img: np.ndarray,
    rois: List[AnyROI],
    detected: Dict[int, Optional[int]],
    answer_key: Dict[int, int],
    kind: str,
) -> np.ndarray:
    """Vẽ kết quả chấm lên ảnh: cam=ROI, đỏ=được tô, xanh=đúng đáp án.

    Màu sắc giúp bạn nhìn nhanh trạng thái từng ô vì dễ phân biệt trực quan.
    """
    out = img.copy()
    log.debug(f"[Draw] kind={kind}, rois={len(rois)}")

    if kind == "circle":
        for idx, r_any in enumerate(rois):
            r: CircleROI = r_any  # type: ignore
            color = (255, 140, 0)
            picked = detected.get(r.question)
            reason = "roi"
            if picked == r.option:
                color = (0, 0, 255)
                reason = "picked"
                if answer_key.get(r.question) == r.option:
                    color = (0, 255, 0)
                    reason = "correct"
            log.debug(
                f"[Draw] circle[{idx}] q={r.question}, opt={r.option}, "
                f"picked={picked}, key={answer_key.get(r.question)}, "
                f"color={color}, reason={reason}"
            )
            cv.circle(out, (r.cx, r.cy), r.r, color, CIRCLE_STROKE)
    else:
        for idx, r_any in enumerate(rois):
            r: ROI = r_any  # type: ignore
            cx = int((r.x_left + r.x_right) / 2)
            cy = int((r.y_top + r.y_bottom) / 2)
            radius = int(
                0.45 * min(r.x_right - r.x_left, r.y_bottom - r.y_top)
            )
            color = (255, 140, 0)
            picked = detected.get(r.question)
            reason = "roi"
            if picked == r.option:
                color = (0, 0, 255)
                reason = "picked"
                if answer_key.get(r.question) == r.option:
                    color = (0, 255, 0)
                    reason = "correct"
            log.debug(
                f"[Draw] rect[{idx}] q={r.question}, opt={r.option}, "
                f"picked={picked}, key={answer_key.get(r.question)}, "
                f"center=({cx},{cy}), radius={radius}, color={color}, "
                f"reason={reason}"
            )
            cv.circle(out, (cx, cy), radius, color, CIRCLE_STROKE)
    return out


# ========== Grading ==========
def grade_exam(
    detected: Dict[int, Optional[int]],
    answer_key: Dict[int, int],
) -> float:
    """Tính điểm thang 10 theo tỷ lệ đúng/số câu.

    Tôi dùng thang điểm 10 vì quen thuộc với giáo dục VN và dễ hiểu.
    """
    if not answer_key:
        log.debug("[Grade] empty answer_key -> 0")
        return 0.0

    details = {q: (detected.get(q), ans) for q, ans in answer_key.items()}
    correct = sum(1 for _q, (pred, ans) in details.items() if pred == ans)
    wrong = [
        (q, pred, ans)
        for q, (pred, ans) in details.items()
        if pred is not None and pred != ans
    ]
    missing = [q for q, (pred, _ans) in details.items() if pred is None]

    log.debug(
        f"[Grade] correct={correct}/{len(answer_key)}, "
        f"wrong={len(wrong)}, missing={len(missing)}"
    )
    if wrong:
        log.debug(f"[Grade] wrong_samples={wrong[:10]}")
    if missing:
        log.debug(f"[Grade] missing_samples={missing[:10]}")

    return correct / len(answer_key) * 10.0


if __name__ == "__main__":
    # Demo grading: đọc input.jpg, threshold, load ROI, detect & chấm điểm demo.
    input_path = "samples/input.jpg"
    roi_path = ROI_CALIB_FILE
    out_dir = "debug_grading"

    safe_mkdir(out_dir)

    img = cv.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    H, W = img.shape[:2]

    if not os.path.exists(roi_path):
        raise FileNotFoundError(
            f"ROI JSON not found: {roi_path}. "
            f"Bạn hãy tạo file ROI trước."
        )

    rois_any, kind = load_any_rois(roi_path, W, H)
    log.info(f"[grading main] Loaded {len(rois_any)} {kind} ROIs")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = build_threshold(gray, method="adaptive_gaussian")  # vì adaptive thường phù hợp OMR hơn global

    detected = detect_marked_generic(
        thresh,
        rois_any,
        kind=kind,
        fill_threshold=0.35,
    )
    log.info(f"[grading main] detected={detected}")

    # Sinh answer_key demo ngẫu nhiên vì bạn chưa gắn file đáp án thật.
    import random

    random.seed(42)
    q_ids = {
        (r.question if isinstance(r, CircleROI) else r.question)
        for r in rois_any
    }
    n_q = max(q_ids) if q_ids else 0
    n_opts = max((r.option for r in rois_any), default=0) + 1 if rois_any else 0
    answer_key = {i + 1: random.randint(0, n_opts - 1) for i in range(n_q)}

    score = grade_exam(detected, answer_key)
    log.info(f"[grading main] Score demo: {score:.2f}/10")

    img_color = img.copy()
    graded = draw_grading_result_generic(
        img_color,
        rois_any,
        detected,
        answer_key,
        kind=kind,
    )
    out_file = f"{out_dir}/final_marked.png"
    cv.imwrite(out_file, graded)
    log.info(f"[grading main] Saved {out_file}")
