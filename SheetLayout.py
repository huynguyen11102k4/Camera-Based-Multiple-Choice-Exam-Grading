import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class SheetLayout:
    def __init__(self,
                 student_id_left=0.5575,
                 student_id_right=0.6735,
                 student_id_top=0.102,
                 student_id_bottom=0.250,
                 student_id_rows=8,
                 student_id_cols=7,
                 
                 quiz_id_left=0.7195,
                 quiz_id_right=0.8035,
                 quiz_id_top=0.102,
                 quiz_id_bottom=0.250,
                 quiz_id_rows=8,
                 quiz_id_cols=5,
                 
                 q1_10_left=0.293,
                 q1_10_right=0.4225,
                 q1_10_top=0.322,
                 q1_10_bottom=0.588,
                 
                 q11_20_left=0.6445,
                 q11_20_right=0.7745,
                 q11_20_top=0.322,
                 q11_20_bottom=0.588,
                 
                 q21_30_left=0.293,
                 q21_30_right=0.4225,
                 q21_30_top=0.6475,
                 q21_30_bottom=0.9152,
                 
                 q31_40_left=0.6445,
                 q31_40_right=0.7745,
                 q31_40_top=0.6475,
                 q31_40_bottom=0.9152,
                 
                 optionsPerRow=5,
                 roiShrink=0):
        
        self.student_id = (student_id_left, student_id_right, student_id_top, student_id_bottom, student_id_rows, student_id_cols)
        self.quiz_id = (quiz_id_left, quiz_id_right, quiz_id_top, quiz_id_bottom, quiz_id_rows, quiz_id_cols)
        
        self.q1_10 = (q1_10_left, q1_10_right, q1_10_top, q1_10_bottom)
        self.q11_20 = (q11_20_left, q11_20_right, q11_20_top, q11_20_bottom)
        self.q21_30 = (q21_30_left, q21_30_right, q21_30_top, q21_30_bottom)
        self.q31_40 = (q31_40_left, q31_40_right, q31_40_top, q31_40_bottom)
        
        self.optionsPerRow = optionsPerRow
        self.roiShrink = roiShrink

    def buildAllRois(self, W, H):
        all_rois = {
            'student_id': self.buildIdRois(self.student_id, W, H, 'student_id'),
            'quiz_id': self.buildIdRois(self.quiz_id, W, H, 'quiz_id'),
            'questions_1_10': self.buildQuestionRois(self.q1_10, W, H, startQ=1, numQuestions=10),
            'questions_11_20': self.buildQuestionRois(self.q11_20, W, H, startQ=11, numQuestions=10),
            'questions_21_30': self.buildQuestionRois(self.q21_30, W, H, startQ=21, numQuestions=10),
            'questions_31_40': self.buildQuestionRois(self.q31_40, W, H, startQ=31, numQuestions=10)
        }
        return all_rois

    def buildIdRois(self, region, W, H, field_name):
        left, right, top, bottom, num_rows, num_cols = region
        
        x0 = int(W * left)
        x1 = int(W * right)
        y0 = int(H * top)
        y1 = int(H * bottom)
        
        rowHeight = (y1 - y0) / num_rows
        colWidth = (x1 - x0) / num_cols
        
        rois = []
        for r in range(num_rows):
            digit = r
            yTop = int(y0 + r * rowHeight)
            yBottom = int(y0 + (r + 1) * rowHeight)
            dy = int((yBottom - yTop) * self.roiShrink)
            
            for c in range(num_cols):
                column = c
                xLeft = int(x0 + c * colWidth)
                xRight = int(x0 + (c + 1) * colWidth)
                dx = int((xRight - xLeft) * self.roiShrink)
                
                roi = (yTop + dy, yBottom - dy, xLeft + dx, xRight - dx, field_name, digit, column)
                rois.append(roi)
        
        return rois

    def buildQuestionRois(self, region, W, H, startQ, numQuestions):
        """Build ROIs for question regions"""
        left, right, top, bottom = region
        
        x0 = int(W * left)
        x1 = int(W * right)
        y0 = int(H * top)
        y1 = int(H * bottom)
        
        rowHeight = (y1 - y0) / numQuestions
        colWidth = (x1 - x0) / self.optionsPerRow
        
        rois = []
        for r in range(numQuestions):
            quesIdx = r + startQ
            yTop = int(y0 + r * rowHeight)
            yBottom = int(y0 + (r + 1) * rowHeight)
            dy = int((yBottom - yTop) * self.roiShrink)
            
            for o in range(self.optionsPerRow):
                xLeft = int(x0 + o * colWidth)
                xRight = int(x0 + (o + 1) * colWidth)
                dx = int((xRight - xLeft) * self.roiShrink)
                
                roi = (yTop + dy, yBottom - dy, xLeft + dx, xRight - dx, quesIdx, o)
                rois.append(roi)
        
        return rois

    def buildRois(self, W, H):
        rois = []
        rois += self.buildQuestionRois(self.q1_10, W, H, startQ=1, numQuestions=10)
        rois += self.buildQuestionRois(self.q11_20, W, H, startQ=11, numQuestions=10)
        rois += self.buildQuestionRois(self.q21_30, W, H, startQ=21, numQuestions=10)
        rois += self.buildQuestionRois(self.q31_40, W, H, startQ=31, numQuestions=10)
        return rois