import os
import random
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from config import MARKER_CENTERS, FULL_WIDTH, FULL_HEIGHT
from marker_utils import findArucoMarkers, getMainCorners
from warp_utils import warpRegion
from roi_utils import detectIdField, detectMarked
from grading import gradeExam
from debug_utils import drawDebugRoisSmall, drawDebugRois, save_step
from SheetLayout import SheetLayout


def imagePineline(imgPath):
    img = cv.imread(imgPath)

    # B1: Grayscale
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(imgGray, cmap='gray')

    # B2: CLAHE
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgGray)
    plt.subplot(2, 3, 2)
    plt.title("CLAHE")
    plt.imshow(imgCLAHE, cmap='gray')

    # B3: Detect markers
    try:
        marker_dict = findArucoMarkers(imgCLAHE)
    except RuntimeError as e:
        print("[LỖI MARKER]", e)
        return

    # B4: Warp full sheet dựa trên 4 góc
    main_corners = getMainCorners(marker_dict)

    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    plt.subplot(2, 3, 4)
    plt.title("Full Sheet Warped")
    plt.imshow(imgWarped, cmap='gray')

    new_marker_dict = findArucoMarkers(imgWarped)

    REGION_CONFIG = {
        'student_id': {
            'marker_ids': [5, 7, 8, 6],
            'dst_corners': np.array([
                MARKER_CENTERS[5],
                MARKER_CENTERS[7],
                MARKER_CENTERS[8],
                MARKER_CENTERS[6]
            ], dtype="float32")
        },
        'quiz_id': {
            'marker_ids': [7, 9, 10, 8],
            'dst_corners': np.array([
                MARKER_CENTERS[7],
                MARKER_CENTERS[9],
                MARKER_CENTERS[10],
                MARKER_CENTERS[8]
            ], dtype="float32")
        },
        'class_id': {
            'marker_ids': [9, 11, 12, 10],
            'dst_corners': np.array([
                MARKER_CENTERS[9],
                MARKER_CENTERS[11],
                MARKER_CENTERS[12],
                MARKER_CENTERS[10]
            ], dtype="float32")
        },
        'questions_1_10': {
            'marker_ids': [13, 15, 16, 14],
            'dst_corners': np.array([
                MARKER_CENTERS[13],
                MARKER_CENTERS[15],
                MARKER_CENTERS[16],
                MARKER_CENTERS[14]
            ], dtype="float32")
        },
        'questions_11_20': {
            'marker_ids': [15, 17, 18, 16],
            'dst_corners': np.array([
                MARKER_CENTERS[15],
                MARKER_CENTERS[17],
                MARKER_CENTERS[18],
                MARKER_CENTERS[16]
            ], dtype="float32")
        },
        'questions_21_30': {
            'marker_ids': [14, 16, 20, 19],
            'dst_corners': np.array([
                MARKER_CENTERS[14],
                MARKER_CENTERS[16],
                MARKER_CENTERS[20],
                MARKER_CENTERS[19]
            ], dtype="float32")
        },
        'questions_31_40': {
            'marker_ids': [16, 18, 21, 20],
            'dst_corners': np.array([
                MARKER_CENTERS[16],
                MARKER_CENTERS[18],
                MARKER_CENTERS[21],
                MARKER_CENTERS[20]
            ], dtype="float32")
        },
        'id': {
            'marker_ids': [5, 11, 12, 6],
            'dst_corners': np.array([
                MARKER_CENTERS[5],
                MARKER_CENTERS[11],
                MARKER_CENTERS[12],
                MARKER_CENTERS[6]
            ], dtype="float32")
        },
        'questions': {
            'marker_ids': [13, 17, 21, 19],
            'dst_corners': np.array([
                MARKER_CENTERS[13],
                MARKER_CENTERS[17],
                MARKER_CENTERS[21],
                MARKER_CENTERS[19]
            ], dtype="float32")
        }
    }

    # warp theo từng region
    for region_name, config in REGION_CONFIG.items():
        marker_ids = config['marker_ids']
        src_corners = np.array(
            [new_marker_dict[mid]['center'] for mid in marker_ids],
            dtype="float32"
        )
        dst_corners = config['dst_corners']

        M_region = cv.getPerspectiveTransform(src_corners, dst_corners)
        warped_region = cv.warpPerspective(imgWarped, M_region, (FULL_WIDTH, FULL_HEIGHT))
        imgWarped = warped_region

        for mid in new_marker_dict:
            pt = new_marker_dict[mid]['center'].reshape(1, 1, 2).astype(np.float32)
            transformed_pt = cv.perspectiveTransform(pt, M_region)
            new_marker_dict[mid]['center'] = transformed_pt[0][0]

    # Blur + Adaptive threshold + Otsu
    imgBlur = cv.GaussianBlur(imgWarped, (3, 3), 0)
    imgThresh = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 201, 2
    )
    plt.subplot(2, 3, 3)
    plt.title("Adaptive Threshold")
    plt.imshow(imgThresh, cmap='gray')
    save_step(imgThresh, "05_Adaptive_Threshold")

    _, thresh = cv.threshold(imgThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    plt.subplot(2, 3, 5)
    plt.title("Final Thresh")
    plt.imshow(thresh, cmap='gray')
    save_step(thresh, "06_Final_Thresh")

    # Warp ảnh màu để debug
    imgColorWarped = cv.warpPerspective(img, M_main, (FULL_WIDTH, FULL_HEIGHT))

    # B5: build ROIs
    layout = SheetLayout()
    all_rois = layout.buildAllRois(FULL_WIDTH, FULL_HEIGHT)

    print("\n=== DETECTING ID FIELDS (with debug) ===")
    student_id = detectIdField(thresh, all_rois['student_id'], 'student_id', debug=True)
    quiz_id = detectIdField(thresh, all_rois['quiz_id'], 'quiz_id', debug=True)
    class_id = detectIdField(thresh, all_rois['class_id'], 'class_id', debug=True)

    print(f"\n[STUDENT ID] {student_id}")
    print(f"[QUIZ ID] {quiz_id}")
    print(f"[CLASS ID] {class_id}")

    print("\n=== DETECTING ANSWERS (with debug for Q1-3) ===")
    all_answers = {}
    for region_name in ['questions_1_10', 'questions_11_20',
                        'questions_21_30', 'questions_31_40']:
        rois = all_rois[region_name]
        marked = detectMarked(thresh, rois, debug=(region_name == 'questions_1_10'))
        all_answers.update(marked)
        print(f"[{region_name}] Detected: {marked}")

    # B6: Grade exam (random key như cũ)
    answerKey = {i: random.randint(0, 4) for i in range(1, 41)}
    print("\n[ANSWER KEY]", answerKey)
    score = gradeExam(all_answers, answerKey)
    print(f"\n==> Điểm số: {score:.2f}/10")

    # Debug ROI nhỏ
    plt.subplot(2, 3, 6)
    imgDebug = drawDebugRoisSmall(imgWarped, all_rois, all_answers, answerKey)
    plt.title("Debug ROIs Preview")
    plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Debug ROI full
    drawDebugRois(imgWarped, all_rois, all_answers, answerKey)

    return all_answers, student_id, quiz_id, class_id