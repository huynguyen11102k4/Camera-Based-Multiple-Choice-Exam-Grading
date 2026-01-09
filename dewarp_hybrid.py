import cv2 as cv
import numpy as np
from marker_config import REGION_CONFIG, MARKER_CENTERS
from warp_utils import warpRegion


def dewarp_hybrid_regional(img, marker_dict, full_width, full_height, main_corners):
    img_base, M_main = warpRegion(img, main_corners, full_width, full_height)
    print(f"✓ Base image: {full_width}x{full_height}")
    
    img_hybrid = img_base.copy()
    
    regions_to_process = [
        'id_area',
        'questions_1_10',
        'questions_11_20',
        'questions_21_30',
        'questions_31_40'
    ]
    
    processed_count = 0
    skipped_count = 0
    
    for region_name in regions_to_process:
        if region_name not in REGION_CONFIG:
            print(f"{region_name}: Không có config, bỏ qua")
            skipped_count += 1
            continue
        
        region_info = REGION_CONFIG[region_name]
        marker_ids = region_info['marker_ids']
        
        src_corners = []
        dst_corners = []
        
        for idx, marker_id in enumerate(marker_ids):
            if marker_id in marker_dict:
                src_corners.append(marker_dict[marker_id]['center'])
                dst_corners.append(MARKER_CENTERS[marker_id])
            else:
                break
        
        if len(src_corners) < 4:
            print(f"{region_name}: Thiếu markers ({len(src_corners)}/4), bỏ qua")
            skipped_count += 1
            continue
        
        src_corners = np.array(src_corners, dtype="float32")
        dst_corners = np.array(dst_corners, dtype="float32")
        
        min_x = int(np.min(dst_corners[:, 0]))
        max_x = int(np.max(dst_corners[:, 0]))
        min_y = int(np.min(dst_corners[:, 1]))
        max_y = int(np.max(dst_corners[:, 1]))
        
        region_width = max_x - min_x
        region_height = max_y - min_y
        
        if region_width <= 0 or region_height <= 0:
            print(f"{region_name}: Kích thước không hợp lệ, bỏ qua")
            skipped_count += 1
            continue
        
        try:
            region_dst_corners = np.array([
                [0, 0],
                [region_width, 0],
                [region_width, region_height],
                [0, region_height]
            ], dtype="float32")
            
            M_region = cv.getPerspectiveTransform(src_corners, region_dst_corners)
            img_region = cv.warpPerspective(img, M_region, (region_width, region_height))
            
            paste_x1 = max(0, min_x)
            paste_y1 = max(0, min_y)
            paste_x2 = min(full_width, max_x)
            paste_y2 = min(full_height, max_y)
            
            crop_x1 = paste_x1 - min_x
            crop_y1 = paste_y1 - min_y
            crop_x2 = crop_x1 + (paste_x2 - paste_x1)
            crop_y2 = crop_y1 + (paste_y2 - paste_y1)
            
            img_region_cropped = img_region[crop_y1:crop_y2, crop_x1:crop_x2]
            
            img_hybrid[paste_y1:paste_y2, paste_x1:paste_x2] = img_region_cropped
            
            print(f"{region_name}: {region_width}x{region_height} → [{min_x},{min_y}]")
            processed_count += 1
            
        except Exception as e:
            print(f"{region_name}: Lỗi perspective - {e}")
            skipped_count += 1
            continue
    
    return img_hybrid


def dewarp_hybrid_regional_smoothed(img, marker_dict, full_width, full_height, main_corners, blend_margin=20):
    img_base, M_main = warpRegion(img, main_corners, full_width, full_height)
    
    img_hybrid = img_base.copy()

    regions_to_process = [
        'questions_1_10',
        'questions_11_20',
        'questions_21_30',
        'questions_31_40'
    ]
    
    processed_count = 0
    
    for region_name in regions_to_process:
        if region_name not in REGION_CONFIG:
            continue
        
        region_info = REGION_CONFIG[region_name]
        marker_ids = region_info['marker_ids']
        
        src_corners = []
        dst_corners = []
        
        for marker_id in marker_ids:
            if marker_id in marker_dict:
                src_corners.append(marker_dict[marker_id]['center'])
                dst_corners.append(MARKER_CENTERS[marker_id])
            else:
                break
        
        if len(src_corners) < 4:
            continue
        
        src_corners = np.array(src_corners, dtype="float32")
        dst_corners = np.array(dst_corners, dtype="float32")
        
        min_x = int(np.min(dst_corners[:, 0]))
        max_x = int(np.max(dst_corners[:, 0]))
        min_y = int(np.min(dst_corners[:, 1]))
        max_y = int(np.max(dst_corners[:, 1]))
        
        region_width = max_x - min_x
        region_height = max_y - min_y
        
        if region_width <= 0 or region_height <= 0:
            continue
        
        try:
            region_dst_corners = np.array([
                [0, 0],
                [region_width, 0],
                [region_width, region_height],
                [0, region_height]
            ], dtype="float32")
            
            M_region = cv.getPerspectiveTransform(src_corners, region_dst_corners)
            img_region = cv.warpPerspective(img, M_region, (region_width, region_height))
            
            alpha = np.ones((region_height, region_width), dtype=np.float32)
            
            if blend_margin > 0:
                for i in range(blend_margin):
                    fade = i / blend_margin
                    alpha[i, :] *= fade  # Top
                    alpha[region_height-1-i, :] *= fade 
                    alpha[:, i] *= fade  # Left
                    alpha[:, region_width-1-i] *= fade 
            
            paste_x1 = max(0, min_x)
            paste_y1 = max(0, min_y)
            paste_x2 = min(full_width, max_x)
            paste_y2 = min(full_height, max_y)
            
            crop_x1 = paste_x1 - min_x
            crop_y1 = paste_y1 - min_y
            crop_x2 = crop_x1 + (paste_x2 - paste_x1)
            crop_y2 = crop_y1 + (paste_y2 - paste_y1)
            
            img_region_cropped = img_region[crop_y1:crop_y2, crop_x1:crop_x2]
            alpha_cropped = alpha[crop_y1:crop_y2, crop_x1:crop_x2]
            
            base_region = img_hybrid[paste_y1:paste_y2, paste_x1:paste_x2].astype(np.float32)
            region_float = img_region_cropped.astype(np.float32)
            
            if len(img_region_cropped.shape) == 3:
                alpha_3ch = np.stack([alpha_cropped]*3, axis=2)
                blended = alpha_3ch * region_float + (1 - alpha_3ch) * base_region
            else:
                blended = alpha_cropped * region_float + (1 - alpha_cropped) * base_region
            
            img_hybrid[paste_y1:paste_y2, paste_x1:paste_x2] = blended.astype(np.uint8)
            
            print(f"{region_name}: {region_width}x{region_height} (blended)")
            processed_count += 1
            
        except Exception as e:
            print(f"{region_name}: Lỗi - {e}")
            continue

    return img_hybrid
