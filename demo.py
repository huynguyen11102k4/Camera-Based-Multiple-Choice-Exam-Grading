import random
import uuid
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

from SheetLayout import SheetLayout

MARKER_CENTERS = {
    5:  (865.0, 129.0),
    6:  (865.0, 961.5),
    7:  (1506.0, 129.0),
    8:  (1506.0, 961.2),
    9:  (1882.5, 129.0),
    10: (1882.0, 961.8),
    11: (2321.0, 129.0),
    12: (2320.8, 961.8),
    13: (439.0, 1136.2),
    14: (438.8, 2023.8),
    15: (1283.5, 1136.0),
    18: (2106.2, 2024.0),
    16: (1283.0, 2024.2),
    17: (2107.0, 1136.5),
    19: (438.5, 3185.5),
    20: (1283.0, 3186.0),
    21: (2104.5, 3185.8),
}

FULL_WIDTH = 2480
FULL_HEIGHT = 3508

def draw_marker_ids(img, marker_dict, title="Detected Markers with IDs"):
    img_vis = img.copy()

    for marker_id, data in marker_dict.items():
        corners = data['corners'].astype(int)
        center = tuple(np.mean(corners, axis=0).astype(int))

        cv.polylines(img_vis, [corners], True, (0, 255, 0), 2)

        cv.circle(img_vis, center, 6, (255, 0, 0), -1)

        cv.putText(img_vis, str(marker_id),
                   (center[0] - 15, center[1] + 15),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 0), 3, cv.LINE_AA)

    plt.figure(figsize=(10, 14))
    plt.imshow(cv.cvtColor(img_vis, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    
    save_step(img_vis, "Detected_Markers_with_IDs")


def findArucoMarkers(imgGray):
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16H5)
    aruco_params = cv.aruco.DetectorParameters()
    
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(imgGray)
    print(f"Tìm thấy {len(corners) if corners else 0} ArUco markers")
    
    if not corners or len(corners) < 4:
        raise RuntimeError(f"Không tìm thấy đủ markers. Chỉ tìm thấy {len(corners) if corners else 0}")
    
    marker_dict = {}
    for i, corner in enumerate(corners):
        center = np.mean(corner[0], axis=0)
        marker_id = ids[i][0] if ids is not None else i
        marker_dict[marker_id] = {
            'center': center,
            'corners': corner[0]
        }
        print(f"[MARKER] ID: {marker_id}, Center: ({center[0]:.1f}, {center[1]:.1f})")
    
    return marker_dict

def getMainCorners(marker_dict):
    required_ids = [1, 3, 2, 4]
    
    if not all(id in marker_dict for id in required_ids):
        missing = [id for id in required_ids if id not in marker_dict]
        raise RuntimeError(f"Không tìm thấy đủ marker góc chính. Thiếu: {missing}")
    
    tl = marker_dict[1]['center']
    tr = marker_dict[3]['center']
    bl = marker_dict[2]['center']
    br = marker_dict[4]['center']
    
    print(f"[MAIN CORNERS] TL(ID1): {tl}, TR(ID2): {tr}, BR(ID4): {br}, BL(ID3): {bl}")
    
    return np.array([tl, tr, br, bl], dtype="float32")

def warpRegion(img, src_corners, dst_width, dst_height):
    dst = np.array([[0, 0], 
                    [dst_width - 1, 0], 
                    [dst_width - 1, dst_height - 1], 
                    [0, dst_height - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(src_corners, dst)
    warped = cv.warpPerspective(img, M, (dst_width, dst_height))
    return warped, M

def extractCircularROI(thresh, center_x, center_y, radius, debug=False):
    H, W = thresh.shape
    
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), radius, 255, -1)
    
    roi_img = cv.bitwise_and(thresh, thresh, mask=mask)
    
    circle_pixels = np.sum(mask > 0)
    filled_pixels = np.sum((roi_img > 0) & (mask > 0))
    
    if circle_pixels > 0:
        filledRatio = float(filled_pixels) / float(circle_pixels)
    else:
        filledRatio = 0.0
    
    if debug:
        print(f"  Circle at ({center_x}, {center_y}), r={radius}: filled={filled_pixels}/{circle_pixels} = {filledRatio:.3f}")
    
    return filledRatio, mask

def detectIdField(thresh, rois, field_name, fillThreshold=0.75, debug=False):
    H, W = thresh.shape
    columnsMap = {}
    
    for roi in rois:
        if len(roi) != 7:
            continue
        yTop, yBottom, xLeft, xRight, fname, digit, column = roi
        
        center_x = int((xLeft + xRight) / 2)
        center_y = int((yTop + yBottom) / 2)
        radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35)
        
        filledRatio, _ = extractCircularROI(thresh, center_x, center_y, radius, 
                                          debug and column == 0 and digit <= 2)
        columnsMap.setdefault(column, []).append((digit, filledRatio))
    
    id_digits = []
    for col in sorted(columnsMap.keys()):
        options = columnsMap[col]
        if not options:
            id_digits.append('_')
            continue
        
        sortedOptions = sorted(options, key=lambda x: x[1], reverse=True)
        bestDigit, bestFilledRatio = sortedOptions[0]
        
        if debug and col == 0:
            top3 = [(d, f"{r:.3f}") for d, r in sortedOptions[:3]]
            print(f"{field_name} Col{col}: Top3 = {top3}, Best = {bestDigit} ({bestFilledRatio:.3f}), Threshold = {fillThreshold}")
        
        if bestFilledRatio < fillThreshold:
            chosenDigit = '_'
        else:
            if len(sortedOptions) > 1:
                secondFilledRatio = sortedOptions[1][1]
                if secondFilledRatio < 0.9 * bestFilledRatio:
                    chosenDigit = str(bestDigit + 1)
                else:
                    chosenDigit = '?'
            else:
                chosenDigit = str(bestDigit + 1)
        
        id_digits.append(chosenDigit)
    
    return ''.join(id_digits)

def detectMarked(thresh, rois, fillThreshold=0.35, debug=False):
    H, W = thresh.shape
    optionsMap = {}
    
    for roi in rois:
        if len(roi) == 6:
            yTop, yBottom, xLeft, xRight, quesIdx, optionIdx = roi
            center_x = int((xLeft + xRight) / 2)
            center_y = int((yTop + yBottom) / 2)
            radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35)
            
            filledRatio, _ = extractCircularROI(thresh, center_x, center_y, radius, 
                                              debug and quesIdx <= 3)
            optionsMap.setdefault(quesIdx, []).append((optionIdx, filledRatio))
    
    results = {}
    for quesIdx, options in optionsMap.items():
        if not options:
            continue
            
        sortedOptions = sorted(options, key=lambda x: x[1], reverse=True)
        bestOptionIdx, bestFilledRatio = sortedOptions[0]
        
        if debug and quesIdx <= 3:
            top3 = [(opt, f"{ratio:.3f}") for opt, ratio in sortedOptions[:3]]
            print(f"Q{quesIdx}: Top3 = {top3}, Best = {bestOptionIdx} ({bestFilledRatio:.3f}), Threshold = {fillThreshold}")
        
        if bestFilledRatio < fillThreshold:
            chosenOption = None 
        else:
            if len(sortedOptions) > 1:
                secondFilledRatio = sortedOptions[1][1]
                if secondFilledRatio < 0.9 * bestFilledRatio:
                    chosenOption = bestOptionIdx
                else:
                    chosenOption = None 
            else:
                chosenOption = bestOptionIdx 
        
        results[quesIdx] = chosenOption
    
    return results

def gradeExam(detectedAnswers, answerKey):
    totalQuestions = len(answerKey)
    correctCount = 0
    for quesIdx, correctOption in answerKey.items():
        detectedOption = detectedAnswers.get(quesIdx, None)
        if detectedOption == correctOption:
            correctCount += 1
    
    print(f"Detected: {detectedAnswers}")
    print(f"Correct: {correctCount}/{totalQuestions}")
    score = (correctCount / totalQuestions) * 10
    return score

def drawDebugRoisSmall(img, all_rois, detectedAnswers=None, answerKey=None):
    imgDebug = cv.resize(img, (img.shape[1]//3, img.shape[0]//3))
    scale = 1/3
    
    for region_name in ['questions_1_10', 'questions_11_20', 'questions_21_30', 'questions_31_40']:
        if region_name in all_rois:
            rois = all_rois[region_name]
            for roi in rois:
                if len(roi) == 6:
                    yTop, yBottom, xLeft, xRight, quesIdx, optionIdx = roi
                    center_x = int((xLeft + xRight) / 2 * scale)
                    center_y = int((yTop + yBottom) / 2 * scale)
                    radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35 * scale)

                    color = (128, 128, 128)
                    thickness = 1
                    
                    if detectedAnswers and answerKey:
                        chosenOption = detectedAnswers.get(quesIdx, None)
                        correctOption = answerKey.get(quesIdx, None)
                        
                        if chosenOption == optionIdx:
                            color = (0, 255, 0) if chosenOption == correctOption else (0, 0, 255)
                            thickness = 2
                    
                    cv.circle(imgDebug, (center_x, center_y), radius, color, thickness)
    
    return imgDebug

def drawDebugRois(img, all_rois, detectedAnswers=None, answerKey=None):
    imgDebug = img.copy()
    
    for region_name in ['questions_1_10', 'questions_11_20', 'questions_21_30', 'questions_31_40']:
        if region_name in all_rois:
            rois = all_rois[region_name]
            for roi in rois:
                if len(roi) == 6:
                    yTop, yBottom, xLeft, xRight, quesIdx, optionIdx = roi
                    center_x = int((xLeft + xRight) / 2)
                    center_y = int((yTop + yBottom) / 2)
                    radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35)

                    color = (128, 128, 128)
                    thickness = 2
                    
                    if detectedAnswers is not None and answerKey is not None:
                        chosenOption = detectedAnswers.get(quesIdx, None)
                        correctOption = answerKey.get(quesIdx, None)
                        
                        if chosenOption == optionIdx:
                            if chosenOption == correctOption:
                                color = (0, 255, 0)
                                thickness = 3
                            else:
                                color = (0, 0, 255)
                                thickness = 3
                        elif optionIdx == correctOption and chosenOption != correctOption:
                            color = (255, 255, 0)
                            thickness = 2
                    
                    cv.circle(imgDebug, (center_x, center_y), radius, color, thickness)
    
    for field_name in ['student_id', 'quiz_id', 'class_id']:
        if field_name in all_rois:
            rois = all_rois[field_name]
            for roi in rois:
                if len(roi) == 7:
                    yTop, yBottom, xLeft, xRight, fname, digit, column = roi
                    center_x = int((xLeft + xRight) / 2)
                    center_y = int((yTop + yBottom) / 2)
                    radius = int(min(yBottom - yTop, xRight - xLeft) * 0.4)
                    
                    cv.circle(imgDebug, (center_x, center_y), radius, (200, 200, 200), 2)
    
    plt.figure(figsize=(14, 10))
    plt.title("Debug ROIs - All Regions")
    plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    # save_step(imgDebug, "07_Debug_ROIs_All_Regions")
    
    return imgDebug

def imagePineline(imgPath):
    img = cv.imread(imgPath)
    
    # B1: Convert to Grayscale
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(imgGray, cmap='gray')
    
    # save_step(imgGray, "01_Original_Grayscale")
        
    # B2: CLAHE & Preprocessing
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgGray)
    plt.subplot(2, 3, 2)
    plt.title("CLAHE")
    plt.imshow(imgCLAHE, cmap='gray')
    
    # save_step(imgCLAHE, "02_CLAHE")
    
    # draw_marker_ids(img, findArucoMarkers(imgCLAHE), title="Detected Markers after CLAHE")
    
    # imgBlur = cv.GaussianBlur(imgCLAHE, (3, 3), 0)
    # imgThresh = cv.adaptiveThreshold(imgCLAHE, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv.THRESH_BINARY, 201, 2)
    # plt.subplot(2, 3, 3)
    # plt.title("Adaptive Threshold")
    # plt.imshow(imgThresh, cmap='gray')
    
    # B3: Detect markers
    try:
        marker_dict = findArucoMarkers(imgCLAHE)
    except RuntimeError as e:
        print("[LỖI MARKER]", e)
        return
    
    # draw_marker_ids(img, marker_dict)
    
    # B4: Warp toàn bộ tờ giấy dựa trên 4 góc chính
    main_corners = getMainCorners(marker_dict)
    FULL_WIDTH = 2480
    FULL_HEIGHT = 3508
    
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    plt.subplot(2, 3, 4)
    plt.title("Full Sheet Warped")
    plt.imshow(imgWarped, cmap='gray')
    
    # save_step(imgWarped, "03_Full_Sheet_Warped")
    
    new_marker_dict = findArucoMarkers(imgWarped)
            
    REGION_CONFIG = {
        'student_id': {
            'marker_ids': [5, 7, 8, 6],
            'src_corners': np.array([
                new_marker_dict[5]['center'],
                new_marker_dict[7]['center'],
                new_marker_dict[8]['center'],
                new_marker_dict[6]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[5],
                MARKER_CENTERS[7],
                MARKER_CENTERS[8],
                MARKER_CENTERS[6]
            ], dtype="float32")
        },
        'quiz_id': {
            'marker_ids': [7, 9, 10, 8],
            'src_corners': np.array([
                new_marker_dict[7]['center'],
                new_marker_dict[9]['center'],
                new_marker_dict[10]['center'],
                new_marker_dict[8]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[7],
                MARKER_CENTERS[9],
                MARKER_CENTERS[10],
                MARKER_CENTERS[8]
            ], dtype="float32")
        },
        'class_id': {
            'marker_ids': [9, 11, 12, 10],
            'src_corners': np.array([
                new_marker_dict[9]['center'],
                new_marker_dict[11]['center'],
                new_marker_dict[12]['center'],
                new_marker_dict[10]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[9],
                MARKER_CENTERS[11],
                MARKER_CENTERS[12],
                MARKER_CENTERS[10]
            ], dtype="float32")
        },
        'questions_1_10': {
            'marker_ids': [13, 15, 16, 14],
            'src_corners': np.array([
                new_marker_dict[13]['center'],
                new_marker_dict[15]['center'],
                new_marker_dict[16]['center'],
                new_marker_dict[14]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[13],
                MARKER_CENTERS[15],
                MARKER_CENTERS[16],
                MARKER_CENTERS[14]
            ], dtype="float32")
        },
        'questions_11_20': {
            'marker_ids': [15, 17, 18, 16],
            'src_corners': np.array([
                new_marker_dict[15]['center'],
                new_marker_dict[17]['center'],
                new_marker_dict[18]['center'],
                new_marker_dict[16]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[15],
                MARKER_CENTERS[17],
                MARKER_CENTERS[18],
                MARKER_CENTERS[16]
            ], dtype="float32")
        },
        'questions_21_30': {
            'marker_ids': [14, 16, 20, 19],
            'src_corners': np.array([
                new_marker_dict[14]['center'],
                new_marker_dict[16]['center'],
                new_marker_dict[20]['center'],
                new_marker_dict[19]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[14],
                MARKER_CENTERS[16],
                MARKER_CENTERS[20],
                MARKER_CENTERS[19]
            ], dtype="float32")
        },
        'questions_31_40': {
            'marker_ids': [16, 18, 21, 20],
            'src_corners': np.array([
                new_marker_dict[16]['center'],
                new_marker_dict[18]['center'],
                new_marker_dict[21]['center'],
                new_marker_dict[20]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[16],
                MARKER_CENTERS[18],
                MARKER_CENTERS[21],
                MARKER_CENTERS[20]
            ], dtype="float32")
        },
        'id': {
            'marker_ids': [5, 11, 12, 6],
            'src_corners': np.array([
                new_marker_dict[5]['center'],
                new_marker_dict[11]['center'],
                new_marker_dict[12]['center'],
                new_marker_dict[6]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[5],
                MARKER_CENTERS[11],
                MARKER_CENTERS[12],
                MARKER_CENTERS[6]
            ], dtype="float32")
        },
        'questions': {
            'marker_ids': [13, 17, 21, 19],
            'src_corners': np.array([
                new_marker_dict[13]['center'],
                new_marker_dict[17]['center'],
                new_marker_dict[21]['center'],
                new_marker_dict[19]['center']
            ], dtype="float32"),
            'dst_corners': np.array([
                MARKER_CENTERS[13],
                MARKER_CENTERS[17],
                MARKER_CENTERS[21],
                MARKER_CENTERS[19]
            ], dtype="float32")
        }
    }

    for region_name, config in REGION_CONFIG.items():
        config['src_corners'] = np.array([
            new_marker_dict[mid]['center'] for mid in config['marker_ids']
        ], dtype="float32")
        
        src_corners = config['src_corners']
        dst_corners = config['dst_corners']
        M_region = cv.getPerspectiveTransform(src_corners, dst_corners)
        warped_region = cv.warpPerspective(imgWarped, M_region, (FULL_WIDTH, FULL_HEIGHT))
        imgWarped = warped_region
        
        for mid in new_marker_dict:
            pt = new_marker_dict[mid]['center'].reshape(1, 1, 2).astype(np.float32)
            transformed_pt = cv.perspectiveTransform(pt, M_region)
            new_marker_dict[mid]['center'] = transformed_pt[0][0]
            
    # save_step(imgWarped, "04_Regions_Warped")
    
    imgBlur = cv.GaussianBlur(imgWarped, (3, 3), 0)
    
    imgThresh = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 201, 2)
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
    
    # B5: Sử dụng SheetLayout để tạo ROIs
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
    for region_name in ['questions_1_10', 'questions_11_20', 'questions_21_30', 'questions_31_40']:
        rois = all_rois[region_name]
        marked = detectMarked(thresh, rois, debug=(region_name == 'questions_1_10'))
        all_answers.update(marked)
        print(f"[{region_name}] Detected: {marked}")
    
    # B6: Grade exam
    answerKey = {i: random.randint(0, 4) for i in range(1, 41)}
    print("\n[ANSWER KEY]", answerKey)
    score = gradeExam(all_answers, answerKey)
    print(f"\n==> Điểm số: {score:.2f}/10")
    
    # Debug visualization
    plt.subplot(2, 3, 6)
    imgDebug = drawDebugRoisSmall(imgColorWarped.copy(), all_rois, all_answers, answerKey)
    plt.title("Debug ROIs Preview")
    plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.tight_layout()

    plt.show()
    
    drawDebugRois(imgWarped, all_rois, all_answers, answerKey)
    
    return all_answers, student_id, quiz_id, class_id

def save_step(img, step_name, folder="debug_steps"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{step_name}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(folder, filename)
    cv.imwrite(path, img)
    print("[Saved]", path)
    
if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/img3.png')
    imagePineline(imgPath)