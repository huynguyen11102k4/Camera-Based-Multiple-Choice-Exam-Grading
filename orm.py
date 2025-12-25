# path: omr.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


def safe_mkdir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại để tránh lỗi khi ghi file."""
    os.makedirs(path, exist_ok=True)


def _safe_imwrite(path: str, img: np.ndarray, note: str = "") -> None:
    """Ghi ảnh và log lại đường dẫn để dễ debug."""
    ok = cv.imwrite(path, img)
    if ok:
        if note:
            log.info(f"[OMR Debug] Saved {note} -> {path}")
        else:
            log.info(f"[OMR Debug] Saved -> {path}")
    else:
        log.warning(f"[OMR Debug] Failed to save image -> {path}")


# Ngưỡng tỉ lệ pixel tô để coi 1 bubble là được chọn
DEFAULT_FILL_THRESHOLD = 0.3


@dataclass
class CircleROI:
    """Một ô tròn trên phiếu (bubble).

    - cx, cy: tâm hình tròn (pixel) trong ảnh A4 đã warp.
    - r: bán kính (pixel).
    - question: số thứ tự câu hỏi (1..N).
    - option: chỉ số đáp án (0=A, 1=B, ...).
    """
    cx: int
    cy: int
    r: int
    question: int
    option: int

    @staticmethod
    def from_dict(d: dict) -> "CircleROI":
        return CircleROI(
            cx=int(d["cx"]),
            cy=int(d["cy"]),
            r=int(d["r"]),
            question=int(d["question"]),
            option=int(d["option"]),
        )


def load_circle_rois(path: str) -> List[CircleROI]:
    """Đọc danh sách CircleROI từ JSON do roi_grid_editor_circle.py export."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rois = [CircleROI.from_dict(d) for d in data]
    log.info(f"[OMR] Loaded {len(rois)} circle ROIs from {path}")
    return rois


def binarize_for_omr(a4_img: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    """Nhị phân hóa ảnh A4 → bubble tô thành vùng trắng (255).

    Nếu debug_dir != None sẽ lưu:
    - omr_step1_gray.png
    - omr_step2_gray_eq.png
    - omr_step3_blur.png
    - omr_step4_binary_raw.png
    - omr_step5_binary.png
    """
    # Bước 1: Gray
    gray = cv.cvtColor(a4_img, cv.COLOR_BGR2GRAY)

    if debug_dir is not None:
        safe_mkdir(debug_dir)
        _safe_imwrite(
            os.path.join(debug_dir, "omr_step1_gray.png"),
            gray,
            "step1 gray",
        )

    # Bước 2: CLAHE (tăng tương phản cục bộ)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "omr_step2_gray_eq.png"),
            gray_eq,
            "step2 gray_eq (CLAHE)",
        )

    # Bước 3: Blur nhẹ để giảm noise
    blur = cv.GaussianBlur(gray_eq, (3, 3), 0)

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "omr_step3_blur.png"),
            blur,
            "step3 blur",
        )

    # Bước 4: Adaptive threshold
    block_size = 201
    C = 5

    binary_raw = cv.adaptiveThreshold(
        blur,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        block_size,
        C,
    )

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "omr_step4_binary_raw.png"),
            binary_raw,
            "step4 binary raw (before morph)",
        )

    # Bước 5: Morphology open để xoá noise lấm tấm
    kernel = np.ones((2, 2), np.uint8)
    binary = cv.morphologyEx(binary_raw, cv.MORPH_OPEN, kernel, iterations=1)

    if debug_dir is not None:
        _safe_imwrite(
            os.path.join(debug_dir, "omr_step5_binary.png"),
            binary,
            "step5 binary final (after morph)",
        )

    return binary


def _fill_ratio_circle(binary: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Tính tỉ lệ pixel tô bên trong hình tròn."""

    h, w = binary.shape[:2]

    # Bounding box giới hạn cho hình tròn
    x0 = max(cx - r, 0)
    y0 = max(cy - r, 0)
    x1 = min(cx + r, w - 1)
    y1 = min(cy + r, h - 1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    patch = binary[y0:y1 + 1, x0:x1 + 1]
    ph, pw = patch.shape[:2]

    # Mask hình tròn trong patch
    mask = np.zeros((ph, pw), dtype=np.uint8)
    cv.circle(mask, (cx - x0, cy - y0), r, 255, -1)

    masked = cv.bitwise_and(patch, patch, mask=mask)
    filled = cv.countNonZero(masked)
    total = cv.countNonZero(mask)

    if total == 0:
        return 0.0
    return filled / float(total)


def detect_answers_from_circle_rois(
    binary: np.ndarray,
    circle_rois: List[CircleROI],
    fill_threshold: float = DEFAULT_FILL_THRESHOLD,
) -> List[int]:
    """Đọc đáp án (A..E) từ danh sách CircleROI.

    - Gom circle theo question.
    - Mỗi câu: chọn option có fill_ratio lớn nhất; nếu < threshold là -1.
    """
    if not circle_rois:
        return []

    # Gom circle theo question
    by_q: Dict[int, List[CircleROI]] = {}
    max_q = 0
    for roi in circle_rois:
        max_q = max(max_q, roi.question)
        by_q.setdefault(roi.question, []).append(roi)

    answers: List[int] = [-1] * max_q

    for q in range(1, max_q + 1):
        rois_q = by_q.get(q)
        if not rois_q:
            answers[q - 1] = -1
            continue

        best_opt = -1
        best_val = 0.0

        for roi in rois_q:
            ratio = _fill_ratio_circle(binary, roi.cx, roi.cy, roi.r)
            if ratio > best_val:
                best_val = ratio
                best_opt = roi.option

        if best_val >= fill_threshold:
            answers[q - 1] = best_opt
        else:
            answers[q - 1] = -1

    return answers


def grade_answers(
    detected_answers: List[int],
    answer_key: List[int],
) -> Tuple[int, List[bool]]:
    """So sánh đáp án detect được với answer_key."""

    n = min(len(detected_answers), len(answer_key))
    per_correct: List[bool] = []
    score = 0

    for i in range(n):
        is_correct = detected_answers[i] == answer_key[i]
        per_correct.append(is_correct)
        if is_correct:
            score += 1

    if len(answer_key) > n:
        per_correct.extend([False] * (len(answer_key) - n))

    return score, per_correct


def draw_omr_debug_overlay(
    a4_img: np.ndarray,
    binary: np.ndarray,
    circle_rois: List[CircleROI],
    detected_answers: List[int],
    answer_key: List[int],
    extra_mark_ratio: float = 0.2,
) -> np.ndarray:
    """Tạo overlay debug trên A4:

    - Xanh lá: ô được chọn & đúng.
    - Đỏ: ô được chọn & sai.
    - Vàng: ô không được chọn nhưng ratio vẫn cao (tô mờ/tô nhầm).
    - Xám: ô còn lại.
    """
    vis = a4_img.copy()
    h, w = binary.shape[:2]

    def clamp_point(cx: int, cy: int, r: int) -> Tuple[int, int]:
        cx = int(np.clip(cx, r, max(r, w - r - 1)))
        cy = int(np.clip(cy, r, max(r, h - r - 1)))
        return cx, cy

    for roi in circle_rois:
        q_idx = roi.question - 1
        if q_idx < 0:
            continue

        ans_idx = detected_answers[q_idx] if q_idx < len(detected_answers) else -1
        key_idx = answer_key[q_idx] if q_idx < len(answer_key) else None

        ratio = _fill_ratio_circle(binary, roi.cx, roi.cy, roi.r)

        # Mặc định xám nhạt
        color = (160, 160, 160)
        thickness = 1

        chosen = (ans_idx == roi.option)
        is_correct = chosen and (key_idx is not None) and (ans_idx == key_idx) and ans_idx != -1

        if is_correct:
            color = (0, 255, 0)      # xanh lá
            thickness = 3
        elif chosen and ans_idx != -1:
            color = (0, 0, 255)      # đỏ
            thickness = 3
        else:
            if ratio >= extra_mark_ratio:
                color = (0, 255, 255)  # vàng
                thickness = 2

        cx, cy = clamp_point(roi.cx, roi.cy, roi.r)
        cv.circle(vis, (cx, cy), roi.r, color, thickness)
        cv.circle(vis, (cx, cy), max(1, roi.r // 6), color, -1)

    return vis


def process_omr_sheet_with_circles(
    a4_img: np.ndarray,
    circle_rois: List[CircleROI],
    answer_key: List[int],
    fill_threshold: float = DEFAULT_FILL_THRESHOLD,
    debug_dir: Optional[str] = None,
) -> Dict:
    """Pipeline OMR cho 1 tờ A4 đã warp:

    1) Nhị phân hoá (và lưu step1..5 nếu debug_dir).
    2) Detect đáp án.
    3) Chấm điểm.
    """
    binary = binarize_for_omr(a4_img, debug_dir=debug_dir)
    answers = detect_answers_from_circle_rois(binary, circle_rois, fill_threshold)
    score, per_correct = grade_answers(answers, answer_key)

    return {
        "binary": binary,
        "answers": answers,
        "score": score,
        "per_correct": per_correct,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OMR chấm trên ảnh A4 đã warp sẵn")
    parser.add_argument(
        "--a4",
        dest="a4_path",
        default=os.path.join("debug_markers", "warped_a4.png"),
        help="Đường dẫn ảnh A4 đã warp (output của markers.py)",
    )
    parser.add_argument(
        "--rois",
        dest="rois_path",
        default="circle_rois.json",
        help="JSON ROIs export từ roi_grid_editor_circle.py",
    )
    parser.add_argument(
        "--key",
        dest="key_path",
        default="answer_key.json",
        help="File JSON chứa đáp án đúng (list index 0..Nchoice-1). "
             "Nếu không tồn tại sẽ dùng tạm key toàn B (1).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="debug_markers",
        help="Thư mục lưu ảnh debug (binary + overlay + stages).",
    )

    args = parser.parse_args()

    # Load ảnh A4 đã warp (do markers.py tạo)
    a4 = cv.imread(args.a4_path)
    if a4 is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh A4: {args.a4_path}. "
            f"Hãy chạy markers.py trước để tạo warped_a4.png."
        )

    # Load Circle ROI
    circle_rois = load_circle_rois(args.rois_path)
    if not circle_rois:
        raise RuntimeError(
            f"Không có ROI nào trong {args.rois_path}. "
            f"Hãy Export Circles JSON từ roi_grid_editor_circle.py."
        )

    # Load answer key
    if os.path.exists(args.key_path):
        with open(args.key_path, "r", encoding="utf-8") as f:
            answer_key = json.load(f)
        answer_key = [int(x) for x in answer_key]
        log.info(f"[OMR] Loaded answer key từ {args.key_path} (len={len(answer_key)})")
    else:
        max_q = max((r.question for r in circle_rois), default=0)
        answer_key = [1] * max_q  # key toàn B để test nhanh
        log.warning(
            f"[OMR] Không tìm thấy {args.key_path}, "
            f"dùng tạm answer_key toàn B (1) cho {max_q} câu."
        )

    # Chấm (và lưu các bước binarize vào out_dir)
    result = process_omr_sheet_with_circles(
        a4_img=a4,
        circle_rois=circle_rois,
        answer_key=answer_key,
        fill_threshold=DEFAULT_FILL_THRESHOLD,
        debug_dir=args.out_dir,
    )
    binary = result["binary"]
    answers = result["answers"]

    print(json.dumps(
        {
            "answers": result["answers"],
            "score": result["score"],
            "per_correct": result["per_correct"],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # Lưu thêm theo step
    safe_mkdir(args.out_dir)

    # Binary final (trùng nội dung step5 nhưng là output của pipeline)
    bin_path = os.path.join(args.out_dir, "omr_step6_binary_final.png")
    _safe_imwrite(bin_path, binary, "step6 binary final (from process)")

    # Overlay màu trên warped_a4
    overlay = draw_omr_debug_overlay(
        a4_img=a4,
        binary=binary,
        circle_rois=circle_rois,
        detected_answers=answers,
        answer_key=answer_key,
        extra_mark_ratio=0.2,
    )
    overlay_path = os.path.join(args.out_dir, "omr_step7_overlay.png")
    _safe_imwrite(overlay_path, overlay, "step7 overlay (answers color-coded)")

    log.info(f"[OMR] Done. Check debug images in {args.out_dir}")
