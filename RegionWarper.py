# RegionWarper.py
import cv2 as cv
import numpy as np


class RegionWarper:
    """
    Warp từng vùng riêng biệt dựa trên 4 ArUco marker tương ứng.
    Hỗ trợ warp toàn tờ và từng vùng nhỏ (ID, câu hỏi...).
    """
    def __init__(self):
        # Định nghĩa các vùng: [TL, TR, BR, BL] theo ID marker
        self.regions = {
            'full_sheet': {
                'ids': [1, 2, 4, 3],        # Góc chính tờ giấy
                'size': (2480, 3508)        # A4 chuẩn
            },
            'student_id': {
                'ids': [5, 6, 10, 9],
                'size': (600, 300)
            },
            'quiz_id': {
                'ids': [6, 10, 12, 15],
                'size': (300, 300)
            },
            'class_id': {
                'ids': [10, 12, 15, 14],
                'size': (300, 300)
            },
            'questions_1_10': {
                'ids': [5, 6, 10, 9],
                'size': (600, 800)
            },
            'questions_11_20': {
                'ids': [6, 10, 12, 15],
                'size': (600, 800)
            },
            'questions_21_30': {
                'ids': [9, 10, 13, 14],
                'size': (600, 800)
            },
            'questions_31_40': {
                'ids': [10, 12, 15, 14],
                'size': (600, 800)
            }
        }

    def warp_region(self, img, marker_dict, region_name):
        """
        Warp một vùng cụ thể từ ảnh gốc.
        Trả về: (warped_image, perspective_matrix) hoặc (None, None) nếu thiếu marker
        """
        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' không tồn tại. Các vùng hợp lệ: {list(self.regions.keys())}")

        config = self.regions[region_name]
        marker_ids = config['ids']
        dst_w, dst_h = config['size']

        # Kiểm tra đủ 4 marker
        missing = [mid for mid in marker_ids if mid not in marker_dict]
        if missing:
            print(f"[WARN] Thiếu marker cho vùng '{region_name}': {missing}")
            return None, None

        # Lấy tọa độ trung tâm 4 góc theo thứ tự: TL → TR → BR → BL
        src_corners = np.array([
            marker_dict[marker_ids[0]]['center'],
            marker_dict[marker_ids[1]]['center'],
            marker_dict[marker_ids[2]]['center'],
            marker_dict[marker_ids[3]]['center']
        ], dtype=np.float32)

        # Đích: hình chữ nhật chuẩn
        dst_corners = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)

        # Tính ma trận perspective
        M = cv.getPerspectiveTransform(src_corners, dst_corners)
        warped = cv.warpPerspective(img, M, (dst_w, dst_h))

        return warped, M