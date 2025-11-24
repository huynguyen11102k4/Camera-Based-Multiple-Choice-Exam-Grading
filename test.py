import random
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

from SheetLayout import SheetLayout

def findArucoMarkers(imgGray):
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_params = cv.aruco.DetectorParameters()
    
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(imgGray)
    print(f"Tìm thấy {len(corners) if corners else 0} ArUco markers")
    
    if corners:
        cv.aruco.drawDetectedMarkers(imgGray, corners, ids)
        plt.subplot(2, 3, 5)
        plt.title(f"Detected {len(corners)} ArUco Markers")
        plt.imshow(cv.cvtColor(imgGray, cv.COLOR_BGR2RGB))
    
    if not corners or len(corners) < 4:
        raise RuntimeError(f"Không tìm thấy đủ 4 ArUco markers. Chỉ tìm thấy {len(corners) if corners else 0}")
    
    marker_centers = []
    for i, corner in enumerate(corners):
        center = np.mean(corner[0], axis=0)
        marker_id = ids[i][0] if ids is not None else i
        marker_centers.append({
            'id': marker_id,
            'center': center,
            'corners': corner[0]
        })
        print(f"[MARKER] ID: {marker_id}, Center: ({center[0]:.1f}, {center[1]:.1f})")
    
    centers = np.array([m['center'] for m in marker_centers])
    
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1).ravel()
    
    idx_tl = np.argmin(s)
    idx_br = np.argmax(s)
    idx_tr = np.argmin(diff)
    idx_bl = np.argmax(diff)

    tl = marker_centers[idx_tl]['center']
    tr = marker_centers[idx_tr]['center']
    br = marker_centers[idx_br]['center']
    bl = marker_centers[idx_bl]['center']
    
    print(f"[CORNERS] TL: {tl}, TR: {tr}, BR: {br}, BL: {bl}")
    
    return np.array([tl, tr, br, bl], dtype="float32")

    

def imagePineline(imgPath):
    img = cv.imread(imgPath)
    
    # B1: Convert to Grayscale & Resize
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    # WIDTH = 500
    # r = WIDTH / float(imgGray.shape[1])
    # dim = (WIDTH, int(imgGray.shape[0] * r))
    # imgResized = cv.resize(imgGray, dim, interpolation=cv.INTER_AREA)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Resized Grayscale Image")
    plt.imshow(imgGray, cmap='gray')

    
    try:
        corners = findArucoMarkers(imgGray)
    except RuntimeError as e:
        print("[LỖI MARKER]", e)
        return
    (tl, tr, br, bl) = corners
    widthA = np.hypot(*(br - bl))
    widthB = np.hypot(*(tr - tl))
    heightA = np.hypot(*(tr - br))
    heightB = np.hypot(*(tl - bl))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    Width=1575
    Height=2316
    
    dst = np.array([[0, 0], 
                    [Width - 1, 0], 
                    [Width - 1, Height - 1], 
                    [0, Height - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(corners, dst)
    imgWarped = cv.warpPerspective(imgGray, M, (Width, Height), flags=cv.INTER_NEAREST)
    plt.subplot(2, 3, 2)
    plt.title("Perspective Transform")
    plt.imshow(imgWarped, cmap='gray')
    
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    imgCLAHE = clahe.apply(imgWarped)
    plt.subplot(2, 3, 3)
    plt.title("CLAHE Image")
    plt.imshow(imgCLAHE, cmap='gray')
    
    imgBlur = cv.GaussianBlur(imgCLAHE, (3, 3), 0)
    plt.subplot(2, 3, 4)
    plt.title("Gaussian Blur Image")
    plt.imshow(imgBlur, cmap='gray')

    imgThresh = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 33, 2)
    plt.subplot(2, 3, 5)
    plt.title("AdaptiveThresholding")
    plt.imshow(imgThresh, cmap='gray')

    _, thresh = cv.threshold(imgThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    plt.subplot(2, 3, 6)
    plt.title("Final Thresholding")
    plt.imshow(thresh, cmap='gray')
    
    plt.show()
    
    layout = SheetLayout()
    rois = layout.buildRois(Width, Height)
    marked = detectMarked(thresh, rois)
    answerKey = [random.randint(0, 4) for _ in range(50)]
    print("Answer Key: ", answerKey)
    score = gradeExam(marked, {i+1: ans for i, ans in enumerate(answerKey)})
    print(f"Điểm số: {score:.2f}/10")

    # Debug
    imgWarpedDebug = cv.warpPerspective(img, M, (Width, Height))
    drawDebugRois(imgWarpedDebug, rois, marked, {i+1: ans for i, ans in enumerate(answerKey)})

def detectMarked(thresh, rois, fillThreshold=0.5):
    H, W = thresh.shape
    results = {}
    optionsMap = {}
    
    for (yTop, yBottom, xLeft, xRight, quesIdx, optionIdx) in rois:
        yTop = max(0, min(H - 1, yTop))
        yBottom = max(0, min(H - 1, yBottom))
        xLeft = max(0, min(W - 1, xLeft))
        xRight = max(0, min(W - 1, xRight))
        roi = thresh[yTop:yBottom, xLeft:xRight]
        filledRatio = float(np.mean(roi>0))
        optionsMap.setdefault(quesIdx, []).append((optionIdx, filledRatio))
    
    for quesIdx, options in optionsMap.items():
        sortedOptions = sorted(options, key=lambda x: x[1], reverse=True)
        bestOptionIdx, bestFilledRatio = sortedOptions[0]
        chosenOption = bestOptionIdx if bestFilledRatio >= fillThreshold else None
        if chosenOption is not None and len(sortedOptions) > 1:
            secondOptionIdx, secondFilledRatio = sortedOptions[1]
            if secondFilledRatio >= fillThreshold:
                chosenOption = None
        results[quesIdx] = chosenOption
    return results

def gradeExam(detectedAnswers, answerKey):
    totalQuestions = len(answerKey)
    correctCount = 0
    for quesIdx, correctOption in answerKey.items():
        detectedOption = detectedAnswers.get(quesIdx, None)
        if detectedOption == correctOption:
            correctCount += 1
    
    print(f"Chosen answers: {detectedAnswers}")
    print(f"Correct answers: {correctCount}/{totalQuestions}")
    score = (correctCount / totalQuestions) * 10
    return score

def drawDebugRois(img, rois, detectedAnswers=None, answerKey=None):
    imgDebug = img.copy()
    for (yTop, yBottom, xLeft, xRight, quesIdx, optionIdx) in rois:
        center_x = int((xLeft + xRight) / 2)
        center_y = int((yTop + yBottom) / 2)

        radius = int(min(yBottom - yTop, xRight - xLeft) * 0.4)

        color = (128, 128, 128)
        thickness = 2
        if detectedAnswers is not None and answerKey is not None:
            chosenOption = detectedAnswers.get(quesIdx, None)
            correctOption = answerKey.get(quesIdx, None)
            if chosenOption == optionIdx:
                if chosenOption == correctOption:
                    color = (0, 255, 255)
                    thickness = 3
                else:
                    color = (0, 0, 255)
                    thickness = 3
            elif optionIdx == correctOption:
                color = (128, 128, 128)
        
        cv.circle(imgDebug, (center_x, center_y), radius, color, thickness)
    
    plt.figure(figsize=(10, 8))
    plt.title("Debug ROI Circles")
    plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return imgDebug

        

if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/testResult.png')
    imagePineline(imgPath)
    
    