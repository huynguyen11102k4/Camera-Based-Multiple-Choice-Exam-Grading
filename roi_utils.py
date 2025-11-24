import cv2 as cv
import numpy as np

from debug_utils import save_step


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
        print(f"  Circle at ({center_x}, {center_y}), r={radius}: "
              f"filled={filled_pixels}/{circle_pixels} = {filledRatio:.3f}")

    return filledRatio, mask


def findCircleCenterByContour(thresh, yTop, yBottom, xLeft, xRight):
    roi = thresh[yTop:yBottom, xLeft:xRight]
    cnts, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_score = 0
    for c in cnts:
        area = cv.contourArea(c)
        if(area < 20):
            continue
        peri = cv.arcLength(c, True)
        if(peri == 0):
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if(circularity > best_score):
            best_score = circularity
            best_cnt = c
    
    if best_cnt is None or best_score < 0.25:
        return None, None
    M = cv.moments(best_cnt)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return xLeft + cx, yTop + cy


def detectIdField(thresh, rois, field_name, fillThreshold=0.75, debug=False):
    H, W = thresh.shape
    columnsMap = {}

    for roi in rois:
        if len(roi) != 7:
            continue
        yTop, yBottom, xLeft, xRight, fname, digit, column = roi

        cx, cy = findCircleCenterByContour(thresh, yTop, yBottom, xLeft, xRight)
        if cx is not None:
            center_x, center_y = cx, cy
        else:
            center_x = int((xLeft + xRight) / 2)
            center_y = int((yTop + yBottom) / 2)

        radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35)

        filledRatio, _ = extractCircularROI(
            thresh, center_x, center_y, radius,
            debug and column == 0 and digit <= 2
        )
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
            print(f"{field_name} Col{col}: Top3 = {top3}, "
                  f"Best = {bestDigit} ({bestFilledRatio:.3f}), "
                  f"Threshold = {fillThreshold}")

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

            cx, cy = findCircleCenterByContour(thresh, yTop, yBottom, xLeft, xRight)
            if cx is not None:
                center_x, center_y = cx, cy
            else:
                center_x = int((xLeft + xRight) / 2)
                center_y = int((yTop + yBottom) / 2)

            radius = int(min(yBottom - yTop, xRight - xLeft) * 0.35)

            filledRatio, _ = extractCircularROI(
                thresh, center_x, center_y, radius,
                debug and quesIdx <= 3
            )
            optionsMap.setdefault(quesIdx, []).append((optionIdx, filledRatio))

    results = {}
    for quesIdx, options in optionsMap.items():
        if not options:
            continue

        sortedOptions = sorted(options, key=lambda x: x[1], reverse=True)
        bestOptionIdx, bestFilledRatio = sortedOptions[0]

        if debug and quesIdx <= 3:
            top3 = [(opt, f"{ratio:.3f}") for opt, ratio in sortedOptions[:3]]
            print(f"Q{quesIdx}: Top3 = {top3}, "
                  f"Best = {bestOptionIdx} ({bestFilledRatio:.3f}), "
                  f"Threshold = {fillThreshold}")

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