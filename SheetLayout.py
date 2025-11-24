import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class SheetLayout:
    def __init__(self,
                 left_ansLeft=0.315,
                 left_ansRight=0.48,
                 left_ansTop=0.36,
                 left_ansBottom=0.975,

                 right_ansLeft=0.535,
                 right_ansRight=0.70,
                 right_ansTop=0.36,
                 right_ansBottom=0.975,

                 questionsPerCol=25,
                 optionsPerRow=5,
                 roiShrink=0.1):
        self.left = (left_ansLeft, left_ansRight, left_ansTop, left_ansBottom)
        self.right = (right_ansLeft, right_ansRight, right_ansTop, right_ansBottom)
        self.questionsPerCol = questionsPerCol
        self.optionsPerRow = optionsPerRow
        self.roiShrink = roiShrink

    def buildRois(self, W, H):
        rois = []
        rois += self._buildHalf(self.left, W, H, startQ=1)
        rois += self._buildHalf(self.right, W, H, startQ=self.questionsPerCol + 1)
        return rois

    def _buildHalf(self, region, W, H, startQ):
        ansLeft, ansRight, ansTop, ansBottom = region
        x0 = int(W * ansLeft)
        x1 = int(W * ansRight)
        y0 = int(H * ansTop)
        y1 = int(H * ansBottom)

        rowHeight = (y1 - y0) / self.questionsPerCol
        colWidth = (x1 - x0)
        optionWidth = colWidth / self.optionsPerRow

        rois = []
        for r in range(self.questionsPerCol):
            quesIdx = r + startQ
            yTop = int(y0 + r * rowHeight)
            yBottom = int(y0 + (r + 1) * rowHeight)
            dy = int((yBottom - yTop) * self.roiShrink)

            for o in range(self.optionsPerRow):
                xLeft = int(x0 + o * optionWidth)
                xRight = int(x0 + (o + 1) * optionWidth)
                dx = int((xRight - xLeft) * self.roiShrink)
                roi = (yTop + dy, yBottom - dy, xLeft + dx, xRight - dx, quesIdx, o)
                rois.append(roi)
        return rois
