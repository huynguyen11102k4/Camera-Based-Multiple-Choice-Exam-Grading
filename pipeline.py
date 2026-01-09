import os
import random
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from config import MARKER_CENTERS, FULL_WIDTH, FULL_HEIGHT
from marker_utils import findArucoMarkers, getMainCorners
from warp_utils import warpRegion
from roi_utils import detectIdField, detectMarked
from grading import gradeExam
from debug_utils import drawDebugRoisSmall, drawDebugRois, save_step
from SheetLayout import SheetLayout
from tps_utils import tps_warp
from mesh_dewarp import mesh_based_warp
from smart_decision import analyze_distortion, print_distortion_report
from dewarp_hybrid import dewarp_hybrid_regional, dewarp_hybrid_regional_smoothed
from adaptive_dewarp import (
    adaptive_dewarp_intelligent, 
    dewarp_adaptive_simple, 
    adaptive_dewarp_hard_selection,
    adaptive_dewarp_regional_smart
)


def imagePineline(imgPath):
    img = cv.imread(imgPath)
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # B1.5: Bilateral filter - giảm nhiễu nhưng giữ cạnh markers
    imgBilateral = cv.bilateralFilter(imgGray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # B2: CLAHE - tăng contrast mạnh hơn cho markers trên giấy nhàu
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgBilateral)

    # B3: Detect markers
    try:
        marker_dict = findArucoMarkers(imgCLAHE)
    except RuntimeError as e:
        print("[LỖI] Không phát hiện đủ markers:", e)
        return

    # B4: Warp full sheet dựa trên 4 góc
    main_corners = getMainCorners(marker_dict)
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    new_marker_dict = findArucoMarkers(imgWarped)
    
    # Phân tích chất lượng ảnh
    analysis = analyze_distortion(new_marker_dict, MARKER_CENTERS)
    print_distortion_report(analysis)
    
    # Lựa chọn phương pháp dewarp dựa trên chất lượng
    dewarp_method = None  # 'mesh', 'tps', 'hybrid', 'hybrid_smooth', hoặc None
    
    if analysis['severity'] == 'GOOD':
        print("\nẢnh chất lượng TỐT - Chỉ cần perspective transform")
        print("Không cần xử lý thêm, ảnh đã phẳng => Xử lý REALTIME")
    else:
        if analysis['severity'] == 'SEVERE':
            print("\nẢNH BỊ BIẾN DẠNG NẶNG - Cần dewarp!")
            print("Lựa chọn:")
            print("  [1] HYBRID - Nhanh (1-3s), perspective từng vùng")
            print("  [2] HYBRID SMOOTH - Nhanh (2-4s), với alpha blending mượt")
            print("  [3] MESH - Nhanh (3-7s), khuyến nghị cho ảnh cong nhiều")
            print("  [4] TPS  - Chính xác hơn nhưng chậm (5-12s)")
            print("  [5] ADAPTIVE BLEND - (3-8s), blend mượt giữa mesh/hybrid")
            print("  [6] ADAPTIVE SHARP - (3-8s), chọn cứng (KHÔNG nhòe)")
            print("  [7] SKIP - Bỏ qua (kết quả kém)")
        else:
            print("\nẢnh biến dạng VỪA PHẢI")
            print("Lựa chọn:")
            print("  [1] HYBRID - Nhanh nhất (1-2s), perspective từng vùng")
            print("  [2] HYBRID SMOOTH - Nhanh (2-3s), với blending mượt")
            print("  [3] MESH - Nhanh (2-5s), đủ tốt cho hầu hết trường hợp")
            print("  [4] TPS  - Chính xác hơn (4-8s)")
            print("  [5] ADAPTIVE BLEND - (2-6s), blend mượt giữa mesh/hybrid")
            print("  [6] ADAPTIVE SHARP - (2-6s), chọn cứng (KHÔNG nhòe)")
            print("  [7] SKIP - Bỏ qua (nhanh nhưng kết quả có thể kém)")
        
        while True:
            choice = input("\nChọn phương pháp (1/2/3/4/5/6/7): ").strip()
            if choice == '1':
                dewarp_method = 'hybrid'
                print("✓ Chọn HYBRID - Perspective từng vùng...")
                break
            elif choice == '2':
                dewarp_method = 'hybrid_smooth'
                print("✓ Chọn HYBRID SMOOTH - Với alpha blending...")
                break
            elif choice == '3':
                dewarp_method = 'mesh'
                print("✓ Chọn MESH - Xử lý nhanh...")
                break
            elif choice == '4':
                dewarp_method = 'tps'
                print("✓ Chọn TPS - Xử lý chính xác...")
                break
            elif choice == '5':
                dewarp_method = 'adaptive_blend'
                print("✓ Chọn ADAPTIVE BLEND - Blend mượt...")
                break
            elif choice == '6':
                dewarp_method = 'adaptive_sharp'
                print("✓ Chọn ADAPTIVE SHARP - Chọn cứng, KHÔNG nhòe...")
                break
            elif choice == '7':
                dewarp_method = None
                print("Bỏ qua dewarp - Kết quả có thể kém")
                break
            else:
                print("Vui lòng nhập 1, 2, 3, 4, 5, 6, hoặc 7")
    
    if dewarp_method in ['mesh', 'tps', 'hybrid', 'hybrid_smooth', 'adaptive_blend', 'adaptive_sharp']:
        if dewarp_method in ['hybrid', 'hybrid_smooth']:
            print(f"\nBắt đầu {dewarp_method.upper()} dewarp...")
            
            if dewarp_method == 'hybrid':
                imgWarped = dewarp_hybrid_regional(
                    imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners
                )
            else:
                imgWarped = dewarp_hybrid_regional_smoothed(
                    imgCLAHE, marker_dict, FULL_WIDTH, FULL_HEIGHT, main_corners, blend_margin=30
                )
            
            print("HYBRID hoàn tất!")
        
        elif dewarp_method == 'adaptive_blend':
            print(f"\nBắt đầu ADAPTIVE BLEND - Blend mượt giữa mesh/hybrid...")
            imgWarped = adaptive_dewarp_intelligent(
                imgCLAHE, marker_dict, main_corners, verbose=True
            )
            print("ADAPTIVE BLEND hoàn tất!")
        
        elif dewarp_method == 'adaptive_sharp':
            print(f"\nBắt đầu ADAPTIVE SHARP - Chọn cứng theo độ lệch marker, KHÔNG nhòe...")
            imgWarped = adaptive_dewarp_regional_smart(
                imgCLAHE, marker_dict, main_corners, threshold_warped=20, verbose=True
            )
            print("ADAPTIVE SHARP hoàn tất!")
        
        else:
            src_pts_dewarp = []
            dst_pts_dewarp = []
            
            for mid in MARKER_CENTERS.keys():
                if mid in new_marker_dict:
                    src_pts_dewarp.append(new_marker_dict[mid]['center'])
                    dst_pts_dewarp.append(MARKER_CENTERS[mid])
            
            src_pts_dewarp = np.array(src_pts_dewarp, dtype="float32")
            dst_pts_dewarp = np.array(dst_pts_dewarp, dtype="float32")
            
            if analysis['severity'] == 'SEVERE':
                smooth_param = 1.5
                downsample = 6
            elif analysis['severity'] == 'MODERATE':
                smooth_param = 0.8
                downsample = 5
            else:
                smooth_param = 0.3
                downsample = 4
            
            if dewarp_method == 'mesh':
                imgWarped = mesh_based_warp(
                    imgWarped, new_marker_dict, MARKER_CENTERS, 
                    smooth=smooth_param, downsample_factor=downsample
                )
                print("MESH hoàn tất!")
            else: 
                imgWarped = tps_warp(
                    imgWarped, src_pts_dewarp, dst_pts_dewarp, 
                    (FULL_WIDTH, FULL_HEIGHT), 
                    smooth=smooth_param
                )
                print("TPS hoàn tất!")
    else:
        print("\nBỏ qua dewarp => Tiếp tục xử lý REALTIME")
    
    # Threshold processing
    imgBilateral2 = cv.bilateralFilter(imgWarped, d=5, sigmaColor=50, sigmaSpace=50)
    imgBlur = cv.GaussianBlur(imgBilateral2, (3, 3), 0)
    
    # Adaptive threshold với blockSize lớn hơn để xử lý bóng từ nếp gấp
    imgThresh = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 251, 5
    )
    
    # Morphological operations để làm sạch noise từ nếp gấp
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    imgThresh = cv.morphologyEx(imgThresh, cv.MORPH_CLOSE, kernel, iterations=1)
    imgThresh = cv.morphologyEx(imgThresh, cv.MORPH_OPEN, kernel, iterations=1)

    _, thresh = cv.threshold(imgThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Morphological cleanup cho final threshold
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel2, iterations=1)

    # Warp ảnh màu để debug
    imgColorWarped = cv.warpPerspective(img, M_main, (FULL_WIDTH, FULL_HEIGHT))

    # B5: build ROIs
    layout = SheetLayout()
    all_rois = layout.buildAllRois(FULL_WIDTH, FULL_HEIGHT)

    # Detect ID fields (không debug)
    student_id = detectIdField(thresh, all_rois['student_id'], 'student_id', debug=False)
    quiz_id = detectIdField(thresh, all_rois['quiz_id'], 'quiz_id', debug=False)

    # Detect answers (không debug)
    all_answers = {}
    for region_name in ['questions_1_10', 'questions_11_20',
                        'questions_21_30', 'questions_31_40']:
        rois = all_rois[region_name]
        marked = detectMarked(thresh, rois, debug=False)
        all_answers.update(marked)

    # B6: Grade exam
    answerKey = {i: random.randint(0, 4) for i in range(1, 41)}
    score = gradeExam(all_answers, answerKey)

    # Hiển thị kết quả
    print(f"\nKẾT QUẢ:")
    print(f"   Mã học sinh: {student_id}")
    print(f"   Mã đề thi: {quiz_id}")
    print(f"   Điểm số: {score:.2f}/10")

    imgDebugFull = drawDebugRois(imgWarped, all_rois, all_answers, answerKey)
    
    plt.figure(figsize=(16, 12))
    plt.imshow(cv.cvtColor(imgDebugFull, cv.COLOR_BGR2RGB))
    plt.title(f"Kết quả chấm điểm: {score:.2f}/10 | Học sinh: {student_id} | Đề: {quiz_id}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return all_answers, student_id, quiz_id