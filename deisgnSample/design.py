# -*- coding: utf-8 -*-
"""
OMR Template (A4 1575x2316) - layout gần giống bản mẫu đính kèm.
- 4 ArUco góc (to) + 8 ArUco nhỏ theo vị trí mẫu
- Cụm thông tin Name/Quiz/Class/Score với gạch ngang
- 3 bảng ID (Student/Quiz/Class) gồm 6 ô nhập + lưới 10x6 bọt số
- 2 cột câu hỏi 1..50 dạng danh sách: "n.  ○A ○B ○C ○D ○E"
- Xuất template.png + layout.json (chứa các ROI vòng tròn)
"""

import os, json
from typing import List, Tuple
import numpy as np
import cv2 as cv

# -------------------- Tham số bố cục chính --------------------
W, H = 1575, 2316                  # A4 @150DPI
MARGIN_OUT = 90                    # lề trắng ngoài
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
ARUCO_SZ_BIG = 140                 # kích thước 4 marker góc
ARUCO_SZ_SMALL = 80                # kích thước các marker nhỏ
FONT = cv.FONT_HERSHEY_SIMPLEX

# Text/graphic sizes
FS_TITLE = 0.75
FS_LABEL = 0.65
FS_SMALL = 0.55
THICK = 2

# Bảng ID
ID_PANEL_W = 420
ID_PANEL_H = 480
ID_PANEL_GAP = 30
ID_BOXES = 6                       # số ô nhập tay
ID_COLS = 6                        # số cột bọt (tương ứng 6 chữ số)
ID_ROWS = 10                       # hàng số 0..9
BUBBLE_R = 15                      # bán kính vòng bọt ID

# --- Fine-tune cho bảng ID ---
ID_TOP_BOX_H     = 36   # chiều cao dải 6 ô nhập tay
ID_TOP_BOX_GAP   = 18   # khoảng cách từ dải ô nhập -> lưới bọt
ID_SIDE_MARGIN   = 28   # lề trái/phải của lưới bọt bên trong panel
ID_BOTTOM_MARGIN = 24   # lề dưới của lưới bọt
BUBBLE_R         = 13   # (giảm nhẹ) bán kính vòng tròn trong bảng ID

# Câu hỏi
OPT_PER_Q = 5
Q_NUM_LEFT_GAP = 32
Q_ROW_H = 36
Q_BUBBLE_R = 13
Q_COL_GAP = 54                     # khoảng cách giữa các bọt trong 1 dòng
Q_LEFT_COL_X = int(W*0.33)         # cột trái vùng bọt
Q_RIGHT_COL_X = int(W*0.58)        # cột phải vùng bọt
Q_Y0 = int(H*0.43)                 # đỉnh khối câu hỏi
Q_BLOCK_H = int(H*0.50)

# -------------------- Công cụ vẽ tiện ích --------------------
def draw_aruco(canvas, center, size, marker_id, label=None):
    x, y = center
    tlx = int(x - size//2); tly = int(y - size//2)
    brx = int(x + size//2); bry = int(y + size//2)
    tlx = max(0, tlx); tly = max(0, tly)
    brx = min(W, brx); bry = min(H, bry)

    marker = np.zeros((size, size), dtype=np.uint8)
    marker = cv.aruco.generateImageMarker(ARUCO_DICT, marker_id, size, marker, 1)
    canvas[tly:bry, tlx:brx] = cv.cvtColor(marker[:bry-tly, :brx-tlx], cv.COLOR_GRAY2BGR)

    if label:
        cv.putText(canvas, label, (tlx, tly-8), FONT, 0.55, (0,0,0), 2, cv.LINE_AA)
    return (tlx, tly, brx, bry)

def put_centered_text(img, text, x, y, scale, color=(0,0,0), thick=THICK):
    (tw, th), _ = cv.getTextSize(text, FONT, scale, thick)
    cv.putText(img, text, (int(x - tw/2), int(y + th/2)), FONT, scale, color, thick, cv.LINE_AA)

# -------------------- Lớp thiết kế --------------------
class QuizTemplateDesigner:
    def __init__(self, width=W, height=H):
        self.W, self.H = width, height
        self.img = np.ones((self.H, self.W, 3), dtype=np.uint8)*255
        self.layout = {
            "image_size": [self.W, self.H],
            "aruco_markers": [],
            "answer_regions": [],
            "id_regions": []
        }

    # --------- Khối thông tin Name/Quiz/Class/Score ---------
    def draw_info_lines(self, x, y, line_w=340, line_gap=42):
        labels = ["Name", "Quiz", "Class", "Score"]
        for i, lb in enumerate(labels):
            yy = y + i*line_gap
            cv.putText(self.img, lb, (x, yy), FONT, FS_LABEL, (0,0,0), THICK, cv.LINE_AA)
            cv.line(self.img, (x+120, yy-6), (x+120+line_w, yy-6), (0,0,0), 2)

    # --------- Bảng ID kiểu mẫu: 6 ô nhập + lưới 10x6 ---------
    def draw_id_panel(self, x0, y0, title="Student ID", key="student_id"):
        # khung & tiêu đề
        cv.rectangle(self.img, (x0, y0), (x0+ID_PANEL_W, y0+ID_PANEL_H), (0,0,0), 2)
        cv.putText(self.img, title, (x0+10, y0-10), FONT, FS_LABEL, (0,0,0), THICK, cv.LINE_AA)

        # 6 ô nhập tay (dải trên)
        box_w = (ID_PANEL_W - 20) // ID_BOXES
        bx_y  = y0 + 10
        for i in range(ID_BOXES):
            bx = x0 + 10 + i*box_w
            cv.rectangle(self.img, (bx, bx_y), (bx + box_w - 6, bx_y + ID_TOP_BOX_H), (0,0,0), 2)

        # lưới bọt 10x6 – nằm dưới dải ô, có lề trái/phải/trên/dưới rõ ràng
        grid_x0 = x0 + ID_SIDE_MARGIN
        grid_y0 = y0 + 10 + ID_TOP_BOX_H + ID_TOP_BOX_GAP
        grid_w  = ID_PANEL_W - ID_SIDE_MARGIN*2
        grid_h  = ID_PANEL_H - (grid_y0 - y0) - ID_BOTTOM_MARGIN

        col_w = grid_w / ID_COLS
        row_h = grid_h / ID_ROWS

        rois = []
        for r in range(ID_ROWS):
            cy = int(grid_y0 + (r + 0.5) * row_h)
            # nhãn 0..9 ở rìa trái trong panel
            cv.putText(self.img, str(r), (x0 + 6, cy + 6), FONT, 0.48, (0,0,0), 1, cv.LINE_AA)
            for c in range(ID_COLS):
                cx = int(grid_x0 + (c + 0.5) * col_w)
                cv.circle(self.img, (cx, cy), BUBBLE_R, (0,0,0), 2, cv.LINE_AA)
                rois.append({"rc":[r,c], "center":[cx,cy], "r":BUBBLE_R})

        self.layout["id_regions"].append({
            "key": key, "title": title, "boxes": ID_BOXES,
            "grid_size":[ID_ROWS, ID_COLS],
            "bubbles": rois,
            "panel_rect":[x0, y0, x0+ID_PANEL_W, y0+ID_PANEL_H]
        })


    # --------- Danh sách câu hỏi dạng "n. ○A ○B ○C ○D ○E" ---------
    def draw_question_block(self, start_q, count, left_x, y0, line_h=Q_ROW_H, key="block"):
        rois = []
        for i in range(count):
            qnum = start_q + i
            yy   = y0 + i*line_h

            # số thứ tự: đo width để chừa khoảng cách trước bọt
            txt = f"{qnum}."
            (tw, th), _ = cv.getTextSize(txt, FONT, FS_SMALL, 1)
            num_x = left_x - (tw + 24)
            num_y = yy + th//2 - 2
            cv.putText(self.img, txt, (num_x, num_y), FONT, FS_SMALL, (0,0,0), 1, cv.LINE_AA)

            # 5 lựa chọn A..E trên 1 hàng
            for o in range(OPT_PER_Q):
                cx = left_x + o*Q_COL_GAP
                cy = yy
                cv.circle(self.img, (cx, cy), Q_BUBBLE_R, (0,0,0), 2, cv.LINE_AA)
                # chữ A..E bên phải vòng
                (aw, ah), _ = cv.getTextSize("A", FONT, 0.5, 1)
                cv.putText(self.img, chr(65+o), (cx + Q_BUBBLE_R + 6, cy + ah//2 - 2),
                        FONT, 0.5, (0,0,0), 1, cv.LINE_AA)
                rois.append([cy - Q_BUBBLE_R, cy + Q_BUBBLE_R,
                            cx - Q_BUBBLE_R, cx + Q_BUBBLE_R,
                            qnum, o])

        self.layout["answer_regions"].append({
            "start_question": start_q,
            "num_questions": count,
            "options_per_question": OPT_PER_Q,
            "rois": rois
        })


    # --------- Cụm ArUco theo mẫu ---------
    def draw_all_aruco(self):
        # 4 góc (to)
        self._add_marker(10, (MARGIN_OUT+ARUCO_SZ_BIG//2, MARGIN_OUT+ARUCO_SZ_BIG//2), ARUCO_SZ_BIG, "TL")
        self._add_marker(11, (self.W-MARGIN_OUT-ARUCO_SZ_BIG//2, MARGIN_OUT+ARUCO_SZ_BIG//2), ARUCO_SZ_BIG, "TR")
        self._add_marker(12, (self.W-MARGIN_OUT-ARUCO_SZ_BIG//2, self.H-MARGIN_OUT-ARUCO_SZ_BIG//2), ARUCO_SZ_BIG, "BR")
        self._add_marker(13, (MARGIN_OUT+ARUCO_SZ_BIG//2, self.H-MARGIN_OUT-ARUCO_SZ_BIG//2), ARUCO_SZ_BIG, "BL")

        # 8 marker nhỏ tham chiếu (tọa độ theo tỉ lệ gần giống bản mẫu)
        pts = [
            (0.17, 0.09), (0.34, 0.09), (0.66, 0.09), (0.83, 0.09),   # dải trên
            (0.33, 0.36), (0.50, 0.36), (0.67, 0.36),                 # trên 2 cột câu
            (0.50, 0.93)                                              # giữa đáy
        ]
        base_id = 24
        for k, (rx, ry) in enumerate(pts):
            self._add_marker(base_id+k, (int(self.W*rx), int(self.H*ry)), ARUCO_SZ_SMALL, None)

    def _add_marker(self, marker_id, center, size, label):
        tlx, tly, brx, bry = draw_aruco(self.img, center, size, marker_id, label)
        self.layout["aruco_markers"].append({
            "id": marker_id,
            "center": [int((tlx+brx)//2), int((tly+bry)//2)],
            "size": size,
            "rect": [tlx, tly, brx, bry],
            **({"label": label} if label else {})
        })

    # --------- Lưu ---------
    def save(self, outdir="output"):
        os.makedirs(outdir, exist_ok=True)
        cv.imwrite(os.path.join(outdir, "template.png"), self.img)
        with open(os.path.join(outdir, "layout.json"), "w", encoding="utf-8") as f:
            json.dump(self.layout, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {outdir}/template.png & layout.json")

# ==================== Build theo mẫu ====================
if __name__ == "__main__":
    d = QuizTemplateDesigner(W, H)

    # ArUco (góc + tham chiếu)
    d.draw_all_aruco()

    # Khối trái: Name/Quiz/Class/Score
    d.draw_info_lines(x=120, y=180, line_w=360, line_gap=48)

    # Cụm 3 bảng ID phía trên giữa
    top_y = 220
    # căn giữa 3 panel
    total_w = 3*ID_PANEL_W + 2*ID_PANEL_GAP
    start_x = (W - total_w)//2
    d.draw_id_panel(start_x + 0*(ID_PANEL_W+ID_PANEL_GAP), top_y, "Student ID", key="student_id")
    d.draw_id_panel(start_x + 1*(ID_PANEL_W+ID_PANEL_GAP), top_y, "Quiz ID",    key="quiz_id")
    d.draw_id_panel(start_x + 2*(ID_PANEL_W+ID_PANEL_GAP), top_y, "Class ID",   key="class_id")

    # Hai khối câu hỏi 1–25, 26–50
    d.draw_question_block(1,  25, Q_LEFT_COL_X,  Q_Y0, key="left")
    d.draw_question_block(26, 25, Q_RIGHT_COL_X, Q_Y0, key="right")

    d.save("output")
