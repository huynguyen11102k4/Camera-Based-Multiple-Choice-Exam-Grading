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
    print(f"[THÔNG TIN] Tìm thấy {len(corners) if corners else 0} ArUco markers")
    
    imgDebug = cv.cvtColor(imgGray, cv.COLOR_GRAY2BGR)
    if corners:
        cv.aruco.drawDetectedMarkers(imgDebug, corners, ids)
        plt.subplot(2, 3, 5)
        plt.title(f"Detected {len(corners)} ArUco Markers")
        plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    
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
        
    # B2: Addaptive histogram equalization & adaptive thresholding
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(48, 48))
    imgCLAHE = clahe.apply(imgGray)
    plt.subplot(2, 3, 2)
    plt.title("CLAHE Image")
    plt.imshow(imgCLAHE, cmap='gray')
    
    imgBlur = cv.GaussianBlur(imgCLAHE, (3, 3), 0)
    plt.subplot(2, 3, 3)
    plt.title("Gaussian Blur Image")
    plt.imshow(imgBlur, cmap='gray')

    imgThresh = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 33, 2)
    plt.subplot(2, 3, 4)
    plt.title("Adaptive Histogram Equalization & Thresholding")
    plt.imshow(imgThresh, cmap='gray')
    
    # B3: Perspective Transform
    try:
        corners = findArucoMarkers(imgCLAHE)
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
    
    dst = np.array([[0, 0], 
                    [maxWidth - 1, 0], 
                    [maxWidth - 1, maxHeight - 1], 
                    [0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(corners, dst)
    imgWarped = cv.warpPerspective(imgBlur, M, (maxWidth, maxHeight))
    warped_blur = cv.GaussianBlur(imgWarped, (3, 3), 0)
    _, thresh = cv.threshold(warped_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    plt.subplot(2, 3, 6)
    plt.title("Warped Image after Perspective Transform")
    plt.imshow(thresh, cmap='gray')
    plt.show()
    
    layout = SheetLayout()
    rois = layout.buildRois(maxWidth, maxHeight)
    marked = detectMarked(thresh, rois)
    answerKey = [random.randint(0, 4) for _ in range(50)]
    print("Answer Key: ", answerKey)
    score = gradeExam(marked, {i+1: ans for i, ans in enumerate(answerKey)})
    print(f"Điểm số: {score:.2f}/10")
    
    drawDebugRois(imgWarped, rois)

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
    print(f"Correct answers: {correctCount} out of {totalQuestions}")
    score = (correctCount / totalQuestions) * 10
    return score

def drawDebugRois(img, rois):
    imgDebug = img.copy()
    for (yTop, yBottom, xLeft, xRight, quesIdx, optionIdx) in rois:
        center_x = int((xLeft + xRight) / 2)
        center_y = int((yTop + yBottom) / 2)

        radius = int(min(yBottom - yTop, xRight - xLeft) * 0.4)

        cv.circle(imgDebug, (center_x, center_y), radius, (0, 255, 255), 2)

        cv.putText(imgDebug, f"{quesIdx}-{optionIdx}",
                   (center_x - 10, center_y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv.LINE_AA)
    
    plt.figure(figsize=(10, 8))
    plt.title("Debug ROI Circles")
    plt.imshow(cv.cvtColor(imgDebug, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return imgDebug

        

if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'testResult.png')
    imagePineline(imgPath)
    
    