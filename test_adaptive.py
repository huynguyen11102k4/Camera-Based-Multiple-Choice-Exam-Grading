"""
Demo script để test adaptive dewarp method
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from config import MARKER_CENTERS, FULL_WIDTH, FULL_HEIGHT
from marker_utils import findArucoMarkers, getMainCorners
from warp_utils import warpRegion
from adaptive_dewarp import (
    define_sheet_regions, 
    analyze_region_warping,
    adaptive_dewarp_intelligent,
    dewarp_adaptive_simple
)
from smart_decision import analyze_distortion, print_distortion_report


def test_adaptive_analysis(imgPath):
    """Test phân tích từng vùng của một ảnh"""
    
    print("="*70)
    print("TEST ADAPTIVE DEWARP - PHÂN TÍCH TỪNG VÙNG")
    print("="*70)
    
    # Load và preprocess
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    imgBilateral = cv.bilateralFilter(imgGray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgBilateral)
    
    # Detect markers
    try:
        marker_dict = findArucoMarkers(imgCLAHE)
        print(f"\n✓ Phát hiện {len(marker_dict)} markers")
    except RuntimeError as e:
        print(f"[LỖI] {e}")
        return
    
    # Warp toàn sheet
    main_corners = getMainCorners(marker_dict)
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    
    # Detect markers trên ảnh warped
    new_marker_dict = findArucoMarkers(imgWarped)
    
    # Phân tích overall
    analysis = analyze_distortion(new_marker_dict, MARKER_CENTERS)
    print_distortion_report(analysis)
    
    # Phân tích từng vùng
    print("\n" + "="*70)
    print("PHÂN TÍCH CHI TIẾT TỪNG VÙNG")
    print("="*70)
    
    regions = define_sheet_regions()
    region_results = {}
    
    for region_name, region_info in regions.items():
        analysis = analyze_region_warping(
            new_marker_dict,
            region_info['markers'],
            MARKER_CENTERS
        )
        region_results[region_name] = analysis
        
        status = "🔴 CONG" if analysis['is_warped'] else "🟢 PHẲNG"
        print(f"\n{region_info['name']:20} {status}")
        print(f"   Markers: {region_info['markers']}")
        print(f"   • Lệch max: {analysis['max_error']:.1f}px")
        print(f"   • Lệch avg: {analysis['avg_error']:.1f}px")
        print(f"   • Điểm cong: {analysis['warping_score']:.1f}/100")
    
    # Thống kê
    warped_regions = [r for r, a in region_results.items() if a['is_warped']]
    flat_regions = [r for r, a in region_results.items() if not a['is_warped']]
    
    print("\n" + "="*70)
    print("TỔNG KẾT")
    print("="*70)
    print(f"\nSố vùng CONG: {len(warped_regions)}/{len(regions)}")
    if warped_regions:
        print(f"   → {', '.join([regions[r]['name'] for r in warped_regions])}")
    
    print(f"\nSố vùng PHẲNG: {len(flat_regions)}/{len(regions)}")
    if flat_regions:
        print(f"   → {', '.join([regions[r]['name'] for r in flat_regions])}")
    
    # Khuyến nghị
    print("\n" + "="*70)
    print("KHUYẾN NGHỊ PHƯƠNG PHÁP")
    print("="*70)
    
    warped_ratio = len(warped_regions) / len(regions)
    
    if warped_ratio == 0:
        print("\n✓ TẤT CẢ VÙNG PHẲNG")
        print("   → Dùng HYBRID (nhanh, 1-2s)")
    elif warped_ratio == 1:
        print("\n✓ TẤT CẢ VÙNG CONG")
        print("   → Dùng MESH (tốt cho cong, 3-7s)")
    else:
        print(f"\n✓ HỖN HỢP ({len(warped_regions)} cong + {len(flat_regions)} phẳng)")
        print("   → Dùng ADAPTIVE (kết hợp thông minh, 3-8s)")
        print(f"   → Hoặc dùng MESH toàn bộ nếu cần đơn giản")
    
    print("\n" + "="*70)
    
    return region_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Dùng ảnh mặc định
        import os
        images_dir = "D:/DemoOMR/images"
        images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            print("Không tìm thấy ảnh trong thư mục images/")
            sys.exit(1)
        
        print("Các ảnh có sẵn:")
        for i, img in enumerate(images, 1):
            print(f"  [{i}] {img}")
        
        choice = input(f"\nChọn ảnh (1-{len(images)}): ").strip()
        try:
            idx = int(choice) - 1
            img_path = os.path.join(images_dir, images[idx])
        except:
            print("Lựa chọn không hợp lệ")
            sys.exit(1)
    
    print(f"\nĐang phân tích: {img_path}\n")
    results = test_adaptive_analysis(img_path)
