import cv2 as cv
import numpy as np

from marker_config import REGION_CONFIG  # vì chứa 9 vùng + dst_corners


def detect_aruco_centers(img):
    """
    Trả về dict {id: (cx, cy)} trên ẢNH BÀI LÀM sau warpRegion,
    vì cần map đúng ID giữa scan và template.
    """
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(img)

    scan_markers = {}
    if ids is not None:
        for c, id_ in zip(corners, ids):
            mid = int(id_[0])
            center = c[0].mean(axis=0)  # trung bình 4 góc vì không phụ thuộc hướng xoay
            scan_markers[mid] = (float(center[0]), float(center[1]))

    return scan_markers


def warp_region_and_paste(img_scan, img_sample, scan_markers, region_cfg):
    """
    Warp 1 vùng từ img_scan sang img_sample theo region_cfg.
    Làm riêng 1 hàm để dùng chung cho cả 9 vùng, vì giúp code gọn & dễ debug.
    """

    marker_ids  = region_cfg["marker_ids"]
    dst_corners = region_cfg["dst_corners"]

    # 1. src_pts = 4 điểm trên scan theo đúng thứ tự marker_ids
    src_pts = np.float32([scan_markers[i] for i in marker_ids])

    # 2. dst_pts_full = 4 điểm global trên sample
    dst_pts_full = dst_corners.astype(np.float32)

    # 3. chuyển dst sang hệ toạ độ local patch
    min_x = int(np.floor(np.min(dst_pts_full[:, 0])))
    min_y = int(np.floor(np.min(dst_pts_full[:, 1])))

    dst_pts_local = dst_pts_full.copy()
    dst_pts_local[:, 0] -= min_x
    dst_pts_local[:, 1] -= min_y
    # vì warpPerspective tạo patch với (0,0) là góc trên trái của patch

    # 4. kích thước patch
    max_x = int(np.ceil(np.max(dst_pts_local[:, 0])))
    max_y = int(np.ceil(np.max(dst_pts_local[:, 1])))
    dst_w = max_x
    dst_h = max_y

    if dst_w <= 0 or dst_h <= 0:
        # vì nếu config sai hoặc marker trùng nhau thì tránh crash
        return img_sample

    # 5. homography từ scan → local patch
    H = cv.getPerspectiveTransform(src_pts, dst_pts_local)

    # 6. warp vùng từ ảnh scan
    patch = cv.warpPerspective(img_scan, H, (dst_w, dst_h))

    # 7. dán patch vào sample tại (min_x, min_y)
    Hs, Ws = img_sample.shape[:2]
    x1, y1 = max(0, min_x), max(0, min_y)
    x2 = min(Ws, x1 + dst_w)
    y2 = min(Hs, y1 + dst_h)

    patch_crop = patch[0:(y2 - y1), 0:(x2 - x1)]
    img_sample[y1:y2, x1:x2] = patch_crop

    return img_sample


def map_scan_to_template(img_scan, template_path):
    """
    Map full sheet (img_scan – đã warpRegion) vào template chuẩn.

    img_scan      : ảnh bài làm sau B4 (warpRegion toàn tờ)
    template_path : đường dẫn tới template (vd /mnt/data/newSample1.png)

    Trả về: ảnh template đã dán nội dung bài làm (img_mapped) –
    vì từ đó trở đi ta chấm trên hệ toạ độ template cho đơn giản.
    """

    img_sample = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if img_sample is None:
        raise RuntimeError(f"Không đọc được template: {template_path}")

    # 1. detect ArUco trên scan
    scan_markers = detect_aruco_centers(img_scan)

    # 2. warp từng vùng theo REGION_CONFIG
    img_out = img_sample.copy()
    for region_name, cfg in REGION_CONFIG.items():
        marker_ids = cfg["marker_ids"]
        if all(mid in scan_markers for mid in marker_ids):
            img_out = warp_region_and_paste(img_scan, img_out, scan_markers, cfg)
        else:
            print(f"[WARN] Bỏ qua vùng {region_name} vì thiếu marker {marker_ids}")

    return img_out
