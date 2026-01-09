import cv2 as cv
import numpy as np
from config import MARKER_CENTERS, FULL_WIDTH, FULL_HEIGHT
from mesh_dewarp import mesh_based_warp
from dewarp_hybrid import dewarp_hybrid_regional
from warp_utils import warpRegion
from marker_utils import findArucoMarkers


def analyze_region_warping(marker_dict, region_markers, marker_centers):
    errors = []
    
    for marker_id in region_markers:
        if marker_id not in marker_dict or marker_id not in marker_centers:
            continue
            
        observed = np.array(marker_dict[marker_id]['center'])
        expected = np.array(marker_centers[marker_id])
        error = np.linalg.norm(observed - expected)
        errors.append(error)
    
    if not errors:
        return {
            'warping_score': 0,
            'max_error': 0,
            'avg_error': 0,
            'is_warped': False
        }
    
    max_error = float(np.max(errors))
    avg_error = float(np.mean(errors))
    
    warping_score = min(100, (max_error * 0.6 + avg_error * 0.4) * 2)
    
    weighted_error = max_error * 0.6 + avg_error * 0.4
    is_warped = (max_error > 25) or (weighted_error > 18) or (max_error > 20 and avg_error > 15)
    
    return {
        'warping_score': warping_score,
        'max_error': max_error,
        'avg_error': avg_error,
        'is_warped': is_warped
    }


def define_sheet_regions():
    from marker_config import REGION_CONFIG
    from SheetLayout import SheetLayout
    
    layout = SheetLayout()
    
    regions = {
        'student_id': {
            'markers': REGION_CONFIG['student_id']['marker_ids'],
            'bounds': (layout.student_id[0], layout.student_id[2], layout.student_id[1], layout.student_id[3]),
            'name': 'Vùng mã học sinh'
        },
        'quiz_id': {
            'markers': REGION_CONFIG['quiz_id']['marker_ids'], 
            'bounds': (layout.quiz_id[0], layout.quiz_id[2], layout.quiz_id[1], layout.quiz_id[3]),
            'name': 'Vùng mã đề thi'
        },
        'questions_1_10': {
            'markers': REGION_CONFIG['questions_1_10']['marker_ids'],
            'bounds': (layout.q1_10[0], layout.q1_10[2], layout.q1_10[1], layout.q1_10[3]),
            'name': 'Câu 1-10'
        },
        'questions_11_20': {
            'markers': REGION_CONFIG['questions_11_20']['marker_ids'],
            'bounds': (layout.q11_20[0], layout.q11_20[2], layout.q11_20[1], layout.q11_20[3]),
            'name': 'Câu 11-20'
        },
        'questions_21_30': {
            'markers': REGION_CONFIG['questions_21_30']['marker_ids'], 
            'bounds': (layout.q21_30[0], layout.q21_30[2], layout.q21_30[1], layout.q21_30[3]),
            'name': 'Câu 21-30'
        },
        'questions_31_40': {
            'markers': REGION_CONFIG['questions_31_40']['marker_ids'],
            'bounds': (layout.q31_40[0], layout.q31_40[2], layout.q31_40[1], layout.q31_40[3]),
            'name': 'Câu 31-40'
        }
    }
    
    return regions


def adaptive_dewarp_intelligent(imgCLAHE, marker_dict, main_corners, 
                                threshold_warped=25, threshold_flat=15,
                                verbose=True):
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    
    from marker_utils import findArucoMarkers
    new_marker_dict = findArucoMarkers(imgWarped)
    
    regions = define_sheet_regions()
    region_analysis = {}
    
    if verbose:
        print("\n" + "="*70)
        print("PHÂN TÍCH ĐỘ CONG TỪNG VÙNG")
        print("="*70)
    
    for region_name, region_info in regions.items():
        analysis = analyze_region_warping(
            new_marker_dict, 
            region_info['markers'], 
            MARKER_CENTERS
        )
        region_analysis[region_name] = analysis
        
        if verbose:
            status = "CONG" if analysis['is_warped'] else "PHẲNG"
            print(f"\n{region_info['name']:20} {status}")
            print(f"   • Lệch max: {analysis['max_error']:.1f}px")
            print(f"   • Lệch avg: {analysis['avg_error']:.1f}px")
            print(f"   • Điểm cong: {analysis['warping_score']:.1f}/100")
    
    warped_regions = [r for r, a in region_analysis.items() if a['is_warped']]
    flat_regions = [r for r, a in region_analysis.items() if not a['is_warped']]
    
    if verbose:
        print("\n" + "="*70)
        print("QUYẾT ĐỊNH CHIẾN LƯỢC DEWARP")
        print("="*70)
    
    if len(warped_regions) == 0:
        if verbose:
            print("\nTẤT CẢ VÙNG PHẲNG → Dùng HYBRID (nhanh, 1-2s)")
        return dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    if len(flat_regions) == 0:
        if verbose:
            print("\nTẤT CẢ VÙNG CONG → Dùng MESH (tốt cho cong, 3-7s)")
        
        src_pts = []
        dst_pts = []
        for mid in MARKER_CENTERS.keys():
            if mid in new_marker_dict:
                src_pts.append(new_marker_dict[mid]['center'])
                dst_pts.append(MARKER_CENTERS[mid])
        
        src_pts = np.array(src_pts, dtype="float32")
        dst_pts = np.array(dst_pts, dtype="float32")
        
        return mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS, 
                              smooth=1.5, downsample_factor=6)
    
    if verbose:
        print(f"\nHỖN HỢP: {len(warped_regions)} vùng cong + {len(flat_regions)} vùng phẳng")
        print("→ ADAPTIVE: Mesh cho vùng cong, Hybrid cho vùng phẳng")
        print(f"\nVùng CONG (dùng Mesh): {', '.join([regions[r]['name'] for r in warped_regions])}")
        print(f"Vùng PHẲNG (dùng Hybrid): {', '.join([regions[r]['name'] for r in flat_regions])}")
    
    src_pts = []
    dst_pts = []
    for mid in MARKER_CENTERS.keys():
        if mid in new_marker_dict:
            src_pts.append(new_marker_dict[mid]['center'])
            dst_pts.append(MARKER_CENTERS[mid])
    
    src_pts = np.array(src_pts, dtype="float32")
    dst_pts = np.array(dst_pts, dtype="float32")
    
    img_mesh = mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS,
                               smooth=1.2, downsample_factor=5)
    
    img_hybrid = dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    result = img_mesh.copy()
    
    H, W = result.shape[:2]
    
    for region_name in flat_regions:
        region_info = regions[region_name]
        bounds = region_info['bounds']
        
        x1 = int(bounds[0] * W)
        y1 = int(bounds[1] * H)
        x2 = int(bounds[2] * W)
        y2 = int(bounds[3] * H)
        
        blend_margin = 5 
        
        core_y1 = y1 + blend_margin
        core_y2 = y2 - blend_margin
        core_x1 = x1 + blend_margin
        core_x2 = x2 - blend_margin
        
        if core_y2 > core_y1 and core_x2 > core_x1:
            result[core_y1:core_y2, core_x1:core_x2] = img_hybrid[core_y1:core_y2, core_x1:core_x2]
        
        mask = np.zeros((H, W), dtype=np.float32)
        
        for i in range(blend_margin):
            fade = (i + 1) / blend_margin 
            
            if y1+i < H and y1+i < core_y1:
                mask[y1+i, x1:x2] = fade
            if y2-i-1 >= 0 and y2-i-1 >= core_y2:
                mask[y2-i-1, x1:x2] = fade
            if x1+i < W and x1+i < core_x1:
                mask[y1:y2, x1+i] = fade
            if x2-i-1 >= 0 and x2-i-1 >= core_x2:
                mask[y1:y2, x2-i-1] = fade
        
        mask_3d = mask[:, :, np.newaxis] if len(result.shape) == 3 else mask
        blended = (result * (1 - mask_3d) + img_hybrid * mask_3d).astype(np.uint8)
        
        blend_mask = (mask > 0).astype(np.uint8)
        if len(result.shape) == 3:
            blend_mask = blend_mask[:, :, np.newaxis]
        result = np.where(blend_mask, blended, result)
    
    if verbose:
        print("\n✓ Hoàn thành adaptive dewarp!")
    
    return result


def dewarp_adaptive_simple(imgCLAHE, marker_dict, main_corners, 
                           overall_severity='MODERATE', verbose=True):
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    
    from marker_utils import findArucoMarkers
    new_marker_dict = findArucoMarkers(imgWarped)
    
    regions = define_sheet_regions()
    warped_count = 0
    
    for region_name, region_info in regions.items():
        analysis = analyze_region_warping(
            new_marker_dict, 
            region_info['markers'], 
            MARKER_CENTERS
        )
        if analysis['is_warped']:
            warped_count += 1
    
    total_regions = len(regions)
    warped_ratio = warped_count / total_regions
    
    if verbose:
        print(f"\nPhân tích: {warped_count}/{total_regions} vùng bị cong ({warped_ratio*100:.0f}%)")
    
    if warped_ratio < 0.3:
        if verbose:
            print("→ Chọn HYBRID (ít vùng cong)")
        return dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    elif warped_ratio > 0.7:
        if verbose:
            print("→ Chọn MESH (nhiều vùng cong)")
        
        src_pts = []
        dst_pts = []
        for mid in MARKER_CENTERS.keys():
            if mid in new_marker_dict:
                src_pts.append(new_marker_dict[mid]['center'])
                dst_pts.append(MARKER_CENTERS[mid])
        
        src_pts = np.array(src_pts, dtype="float32")
        dst_pts = np.array(dst_pts, dtype="float32")
        
        return mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS,
                              smooth=1.5, downsample_factor=6)
    
    else:
        if verbose:
            print("→ Chọn ADAPTIVE (hỗn hợp)")
        return adaptive_dewarp_intelligent(imgCLAHE, marker_dict, main_corners, verbose=False)


def adaptive_dewarp_hard_selection(imgCLAHE, marker_dict, main_corners, verbose=True):

    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    
    from marker_utils import findArucoMarkers
    new_marker_dict = findArucoMarkers(imgWarped)
    
    regions = define_sheet_regions()
    region_analysis = {}
    
    if verbose:
        print("\n" + "="*70)
        print("ADAPTIVE HARD SELECTION - KHÔNG BLEND")
        print("="*70)
    
    for region_name, region_info in regions.items():
        analysis = analyze_region_warping(
            new_marker_dict, 
            region_info['markers'], 
            MARKER_CENTERS
        )
        region_analysis[region_name] = analysis
        
        if verbose:
            status = "CONG => MESH" if analysis['is_warped'] else "PHẲNG => HYBRID"
            print(f"{region_info['name']:20} {status}")
    
    warped_regions = [r for r, a in region_analysis.items() if a['is_warped']]
    flat_regions = [r for r, a in region_analysis.items() if not a['is_warped']]
    
    if verbose:
        print(f"\n{len(warped_regions)} vùng CONG + {len(flat_regions)} vùng PHẲNG")
    
    if len(warped_regions) == 0:
        if verbose:
            print("→ Tất cả phẳng, dùng HYBRID")
        return dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    if len(flat_regions) == 0:
        if verbose:
            print("→ Tất cả cong, dùng MESH")
        src_pts = []
        dst_pts = []
        for mid in MARKER_CENTERS.keys():
            if mid in new_marker_dict:
                src_pts.append(new_marker_dict[mid]['center'])
                dst_pts.append(MARKER_CENTERS[mid])
        
        src_pts = np.array(src_pts, dtype="float32")
        dst_pts = np.array(dst_pts, dtype="float32")
        
        return mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS,
                              smooth=1.2, downsample_factor=5)
    
    if verbose:
        print("→ Tạo Mesh version...")
    
    src_pts = []
    dst_pts = []
    for mid in MARKER_CENTERS.keys():
        if mid in new_marker_dict:
            src_pts.append(new_marker_dict[mid]['center'])
            dst_pts.append(MARKER_CENTERS[mid])
    
    src_pts = np.array(src_pts, dtype="float32")
    dst_pts = np.array(dst_pts, dtype="float32")
    
    img_mesh = mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS,
                               smooth=1.2, downsample_factor=5)
    
    if verbose:
        print("→ Tạo Hybrid version...")
    
    img_hybrid = dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    if verbose:
        print("→ Hard selection từng vùng...")
    
    result = img_mesh.copy()
    H, W = result.shape[:2]
    
    for region_name in flat_regions:
        region_info = regions[region_name]
        bounds = region_info['bounds']
        
        x1 = int(bounds[0] * W)
        y1 = int(bounds[1] * H)
        x2 = int(bounds[2] * W)
        y2 = int(bounds[3] * H)
        
        result[y1:y2, x1:x2] = img_hybrid[y1:y2, x1:x2]
    
    if verbose:
        print("Hoàn thành - Giữ nguyên độ sắc nét!")
    
    return result


def adaptive_dewarp_regional_smart(imgCLAHE, marker_dict, main_corners, 
                                   threshold_warped=20, verbose=True):
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    new_marker_dict = findArucoMarkers(imgWarped)
    
    regions = define_sheet_regions()
    region_analysis = {}
    
    if verbose:
        print("\n" + "="*70)
        print("PHÂN TÍCH ĐỘ LỆCH MARKER TỪNG VÙNG")
        print("="*70)
    
    for region_name, region_info in regions.items():
        analysis = analyze_region_warping(
            new_marker_dict, 
            region_info['markers'], 
            MARKER_CENTERS
        )
        region_analysis[region_name] = analysis
        
        if verbose:
            status = "CONG" if analysis['max_error'] >= threshold_warped else "PHẲNG"
            print(f"\n{region_info['name']:20} {status}")
            print(f"   Markers: {region_info['markers']}")
            print(f"   • Lệch MAX: {analysis['max_error']:.1f}px (ngưỡng: {threshold_warped}px)")
            print(f"   • Lệch AVG: {analysis['avg_error']:.1f}px")
    
    warped_regions = [r for r, a in region_analysis.items() if a['max_error'] >= threshold_warped]
    flat_regions = [r for r, a in region_analysis.items() if a['max_error'] < threshold_warped]
    
    if len(warped_regions) == 0:
        if verbose:
            print("\nTẤT CẢ VÙNG PHẲNG => Dùng HYBRID toàn sheet")
        return dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)
    
    if len(flat_regions) == 0:
        if verbose:
            print("\nTẤT CẢ VÙNG CONG → Dùng MESH toàn sheet")
        return mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS, 
                              smooth=1.5, downsample_factor=6)
    
    if verbose:
        print(f"\nHỖN HỢP: {len(warped_regions)} vùng cong + {len(flat_regions)} vùng phẳng")
        print(f"\nVùng CONG (Mesh): {', '.join([regions[r]['name'] for r in warped_regions])}")
        print(f"Vùng PHẲNG (Hybrid): {', '.join([regions[r]['name'] for r in flat_regions])}")
        print("\n=> Chiến lược: Mesh toàn sheet, thay vùng phẳng bằng Hybrid")
    
    if verbose:
        print("\nTạo Mesh version...")
    
    img_mesh = mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS,
                               smooth=1.2, downsample_factor=5)
    
    if verbose:
        print("Tạo Hybrid version...")
    
    img_hybrid = dewarp_hybrid_regional(imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners)

    
    result = img_mesh.copy()
    H, W = result.shape[:2]
    
    for region_name in flat_regions:
        region_info = regions[region_name]
        bounds = region_info['bounds']
        
        x1 = int(bounds[0] * W)
        y1 = int(bounds[1] * H)
        x2 = int(bounds[2] * W)
        y2 = int(bounds[3] * H)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        result[y1:y2, x1:x2] = img_hybrid[y1:y2, x1:x2]
        
        if verbose:
            print(f"Copy {region_info['name']}: [{x1}:{x2}, {y1}:{y2}]")

    
    return result

