# omr_pipeline_any.py
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

from core import (
    A4_PX,
    DEBUG_DIR,
    ROI_CALIB_FILE,
    TEMPLATE_LAYOUT_FILE,
    log,
    safe_cleanup_dir,
    safe_mkdir,
    show_and_save_step,
)
from grading import detect_marked_generic, draw_grading_result_generic, grade_exam
from markers import (
    auto_crop_using_markers,
    extract_template_marker_layout,
    load_template_marker_layout,
    warp_to_a4,
)
from rois import AnyROI, CircleROI, ROI, interactive_roi_adjustment, load_any_rois
from threshold import ThresholdMethod, build_threshold


# ========== Pipeline ==========
def pipeline(
    img_path: str,
    template_layout: Dict[int, List[float]],
    sheet_layout_builder,
    roi_calib_file: str = ROI_CALIB_FILE,
    debug_dir: str = DEBUG_DIR,
    interactive: bool = True,
    threshold_method: ThresholdMethod = "adaptive_gaussian",
    fill_threshold: float = 0.35,
) -> None:
    """Pipeline đầy đủ từ ảnh quét đến kết quả chấm.

    Tôi tách pipeline thành các bước rõ ràng:
      1. Đọc ảnh gốc
      2. Warp về A4 dùng ArUco
      3. Auto-crop vùng bài
      4. Chuyển xám + threshold (chọn thuật toán)
      5. Load ROI (circle/rect) hoặc build từ layout
      6. Detect ô được tô
      7. Sinh answer_key demo
      8. Grade + vẽ kết quả lên ảnh

    Vì pipeline gọi nhiều module, việc tách giúp bạn test từng bước độc lập.
    """
    log.debug(
        f"[Pipe] start img_path={img_path}, "
        f"template_layout_ids={sorted(list(template_layout.keys()))}, "
        f"debug_dir={debug_dir}, interactive={interactive}, "
        f"thresh_method={threshold_method}"
    )

    safe_mkdir(debug_dir)
    safe_cleanup_dir(debug_dir)

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")
    log.debug(f"[Pipe] original_shape={img.shape}")

    show_and_save_step(1, "original", img, debug_dir, interactive=interactive)

    # Chuẩn hoá theo A4 qua homography
    warped = warp_to_a4(img, template_layout, A4_PX)
    show_and_save_step(2, "warped_A4", warped, debug_dir, interactive=interactive)

    # Cắt gọn vùng bài
    cropped, overlay, bbox = auto_crop_using_markers(warped)
    log.debug(f"[Pipe] crop_bbox={bbox}")
    show_and_save_step(3, "crop_box", overlay, debug_dir, interactive=interactive)
    show_and_save_step(4, "cropped", cropped, debug_dir, interactive=interactive)

    # Nhị phân hoá để đo phần tô
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    log.debug(f"[Pipe] cropped_shape={cropped.shape}, gray_shape={gray.shape}")
    thresh = build_threshold(gray, method=threshold_method)
    show_and_save_step(5, "threshold", thresh, debug_dir, interactive=interactive)

    H, W = cropped.shape[:2]
    log.debug(f"[Pipe] working_size=({W},{H})")

    rois_any: List[AnyROI]
    kind: str

    if os.path.exists(roi_calib_file):
        rois_any, kind = load_any_rois(roi_calib_file, W, H)
        log.info(f"[Pipe] Loaded ROI calibration ({kind}): {roi_calib_file}")
    else:
        # Fallback: dùng layout builder cũ (rect)
        raw_rois = sheet_layout_builder(W, H)
        log.debug(f"[Pipe] built {len(raw_rois)} raw ROIs from layout builder")
        # Chuyển sang ROI dataclass + clamp
        rois_rect: List[ROI] = []
        for r in raw_rois:
            if isinstance(r, ROI):
                rois_rect.append(r.clamp_to(W, H))
            else:
                rois_rect.append(ROI.from_seq(r).clamp_to(W, H))

        if interactive:
            adjusted = interactive_roi_adjustment(
                cropped,
                rois_rect,
                save_path=roi_calib_file,
            )
            if adjusted is not None:
                rois_rect = adjusted
        rois_any = rois_rect
        kind = "rect"

    log.debug(f"[Pipe] ROI kind={kind}, count={len(rois_any)}")

    # Suy ra đáp án được tô
    detected = detect_marked_generic(
        thresh,
        rois_any,
        kind,
        fill_threshold=fill_threshold,
    )
    log.debug(f"[Pipe] detected={detected}")

    # Sinh đáp án mẫu ngẫu nhiên để demo
    random.seed(42)
    q_ids = {
        (r.question if isinstance(r, CircleROI) else r.question)
        for r in rois_any
    }
    n_q = max(q_ids) if q_ids else 50
    n_opts = (
        max((r.option for r in rois_any), default=4) + 1 if rois_any else 5
    )
    answer_key = {i + 1: random.randint(0, n_opts - 1) for i in range(n_q)}
    log.debug(
        f"[Pipe] answer_key(sample 10)="
        f"{dict(list(answer_key.items())[:10])} ... total={len(answer_key)}"
    )

    score = grade_exam(detected, answer_key)
    log.info(f"Score: {score:.2f}/10")

    # Vẽ kết quả cuối
    graded = draw_grading_result_generic(
        cropped,
        rois_any,
        detected,
        answer_key,
        kind,
    )
    show_and_save_step(
        6,
        "final_marked",
        graded,
        debug_dir,
        interactive=interactive,
    )
    out_path = os.path.join(debug_dir, "final_marked.png")
    ok = cv.imwrite(out_path, graded)
    log.info(f"Saved: {out_path} (ok={ok})")


# ========== CLI ==========
def main() -> None:
    """CLI: chạy pipeline với tham số dòng lệnh.

    Tôi giữ CLI để bạn chạy:
        python omr_pipeline_any.py --input Demo6.jpg
    """
    parser = argparse.ArgumentParser(
        description="OMR pipeline (rect/circle ROI auto-detect)."
    )
    parser.add_argument(
        "--template",
        default="template_scan.png",
        help="Template image path",
    )
    parser.add_argument(
        "--input",
        default="Demo6.jpg",
        help="Scanned sheet path",
    )
    parser.add_argument(
        "--template-layout",
        default=TEMPLATE_LAYOUT_FILE,
        help="Template layout JSON",
    )
    parser.add_argument(
        "--roi-calib",
        default=ROI_CALIB_FILE,
        help="ROI JSON (circle or rect)",
    )
    parser.add_argument(
        "--debug-dir",
        default=DEBUG_DIR,
        help="Debug output directory",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable interactive windows",
    )
    parser.add_argument(
        "--thresh-method",
        default="adaptive_gaussian",
        choices=["adaptive_gaussian", "adaptive_mean", "otsu"],
        help="Threshold method",
    )
    parser.add_argument(
        "--fill-threshold",
        type=float,
        default=0.35,
        help="Fill ratio threshold (0..1) to mark a choice as filled",
    )
    args = parser.parse_args()

    log.debug(f"[CLI] args={vars(args)}")

    # Import SheetLayout từ code của bạn
    from SheetLayout import SheetLayout

    if not os.path.exists(args.template_layout):
        extract_template_marker_layout(args.template, args.template_layout)
    template_layout = load_template_marker_layout(args.template_layout)

    layout = SheetLayout()
    sheet_layout_builder = layout.buildRois

    pipeline(
        img_path=args.input,
        template_layout=template_layout,
        sheet_layout_builder=sheet_layout_builder,
        roi_calib_file=args.roi_calib,
        debug_dir=args.debug_dir,
        interactive=not args.no_ui,
        threshold_method=args.thresh_method,  # cho phép chọn thuật toán
        fill_threshold=args.fill_threshold,
    )


if __name__ == "__main__":
    main()
