import os
import uuid
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def save_step(img, step_name, folder="debug_steps"):
    pass

def findCircleCenterByContour(thresh, yTop, yBottom, xLeft, xRight):
    roi = thresh[yTop:yBottom, xLeft:xRight]
    cnts, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_score = 0
    for c in cnts:
        area = cv.contourArea(c)
        if area < 20:
            continue
        peri = cv.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > best_score:
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

def drawDebugRoisSmall(img, all_rois, detectedAnswers=None, answerKey=None):
    imgDebug = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    scale = 1 / 3

    for region_name in ['questions_1_10', 'questions_11_20',
                        'questions_21_30', 'questions_31_40']:
        if region_name in all_rois:
            rois = all_rois[region_name]
            for roi in rois:
                if len(roi) == 6:
                    yTop, yBottom, xLeft, xRight, quesIdx, optionIdx = roi
                    cx, cy = findCircleCenterByContour(img, yTop, yBottom, xLeft, xRight)
                    if cx is not None:
                        center_x, center_y = cx, cy
                    else:
                        center_x = int((xLeft + xRight) / 2 * scale)
                        center_y = int((yTop + yBottom) / 2 * scale)
                    radius = int(min(yBottom - yTop, xRight - xLeft)
                                 * 0.35 * scale)

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

    for region_name in ['questions_1_10', 'questions_11_20',
                        'questions_21_30', 'questions_31_40']:
        if region_name in all_rois:
            rois = all_rois[region_name]
            for roi in rois:
                if len(roi) == 6:
                    yTop, yBottom, xLeft, xRight, quesIdx, optionIdx = roi
                    cx, cy = findCircleCenterByContour(img, yTop, yBottom, xLeft, xRight)
                    if cx is not None:
                        center_x, center_y = cx, cy
                    else:
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

    for field_name in ['student_id', 'quiz_id']:
        if field_name in all_rois:
            rois = all_rois[field_name]
            for roi in rois:
                if len(roi) == 7:
                    yTop, yBottom, xLeft, xRight, fname, digit, column = roi
                    center_x = int((xLeft + xRight) / 2)
                    center_y = int((yTop + yBottom) / 2)
                    radius = int(min(yBottom - yTop, xRight - xLeft) * 0.4)

                    cv.circle(imgDebug, (center_x, center_y), radius, (200, 200, 200), 2)

    return imgDebug
    
