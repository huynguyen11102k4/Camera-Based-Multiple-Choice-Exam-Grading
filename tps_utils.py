import numpy as np
import cv2 as cv
from scipy.interpolate import Rbf

# def tps_kernel(r):
#     r = np.maximum(r, 1e-12)
#     return (r*r) * np.log(r*r)

def tps_warp(img, src_points, dst_points, dst_size, smooth=0.5):
    """
    TPS warp với smooth parameter để xử lý giấy bị nếp gấp mạnh
    
    Args:
        img: Ảnh nguồn
        src_points: Các điểm control trên ảnh nguồn
        dst_points: Các điểm mục tiêu
        dst_size: Kích thước ảnh đích (width, height)
        smooth: Tham số làm mượt (0.5-2.0). Giá trị cao hơn = mượt hơn, phù hợp với giấy nhàu
    """
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    W, H = dst_size

    N = src_points.shape[0]
    if N < 3:
        print("[WARN] TPS cần ít nhất 3 điểm")
        return img

    src_x, src_y = src_points[:, 0], src_points[:, 1]
    dst_x, dst_y = dst_points[:, 0], dst_points[:, 1]

    # Nội suy trường độ lệch dx, dy từ source => dest
    # dest = source + delta  =>  delta = dest - source
    dx = dst_x - src_x
    dy = dst_y - src_y

    # Khởi tạo Rbf với kernel thin_plate để mô hình hoá dx, dy
    # smooth điều chỉnh độ mượt - quan trọng cho giấy nhàu nát
    rbf_dx = Rbf(src_x, src_y, dx, function="thin_plate", smooth=smooth)
    rbf_dy = Rbf(src_x, src_y, dy, function="thin_plate", smooth=smooth)

    # Tạo lưới toạ độ trên ảnh đích u, v với grid dày hơn
    # Grid dày hơn giúp xử lý biến dạng cục bộ tốt hơn
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

    # Tính độ lệch tại từng điểm u, v trên ảnh đích
    delta_x = rbf_dx(grid_x, grid_y)
    delta_y = rbf_dy(grid_x, grid_y)
    
    # src = dest - delta
    source_map_x = grid_x - delta_x
    source_map_y = grid_y - delta_y

    h_src, w_src = img.shape[:2]
    source_map_x = np.clip(source_map_x, 0, w_src - 1)
    source_map_y = np.clip(source_map_y, 0, h_src - 1)

    map_x = source_map_x.astype(np.float32)
    map_y = source_map_y.astype(np.float32)

    # Warp ảnh bằng nội suy INTER_CUBIC cho chất lượng tốt hơn với giấy nhàu
    img_tps = cv.remap(img, map_x, map_y, interpolation=cv.INTER_CUBIC, 
                       borderMode=cv.BORDER_REPLICATE)

    return img_tps
