import cv2 as cv
import numpy as np
from config import MARKER_CENTERS


def analyze_distortion(marker_dict_warped, expected_centers):
    errors = []
    problem_markers = []
    
    for marker_id in marker_dict_warped:
        if marker_id not in expected_centers:
            continue
        
        observed = np.array(marker_dict_warped[marker_id]['center'])
        expected = np.array(expected_centers[marker_id])
        
        error = np.linalg.norm(observed - expected)
        errors.append(error)
        
        if error > 20:
            problem_markers.append({
                'id': marker_id,
                'error': float(error),
                'observed': observed.tolist(),
                'expected': expected.tolist()
            })
    
    if not errors:
        return {
            'max_error': 0,
            'avg_error': 0,
            'severity': 'UNKNOWN',
            'problem_areas': [],
            'num_markers': 0
        }
    
    max_error = float(np.max(errors))
    avg_error = float(np.mean(errors))
    
    # Phân loại độ nghiêm trọng
    if max_error < 15 and avg_error < 8:
        severity = 'GOOD'
    elif max_error < 40 and avg_error < 20:
        severity = 'MODERATE'
    else:
        severity = 'SEVERE'
    
    return {
        'max_error': max_error,
        'avg_error': avg_error,
        'severity': severity,
        'problem_areas': problem_markers,
        'num_markers': len(errors)
    }


def print_distortion_report(analysis):
    """In báo cáo phân tích biến dạng (chỉ thông tin, không hỏi)"""
    print("\n" + "="*70)
    print("PHÂN TÍCH CHẤT LƯỢNG GIẤY")
    print("="*70)
    
    print(f"\nĐộ lệch markers sau perspective transform:")
    print(f"   • Lệch tối đa: {analysis['max_error']:.1f} pixels")
    print(f"   • Lệch trung bình: {analysis['avg_error']:.1f} pixels")
    print(f"   • Số markers kiểm tra: {analysis['num_markers']}")
    
    severity = analysis['severity']
    
    if severity == 'GOOD':
        print(f"\nCHẤT LƯỢNG TỐT → XỬ LÝ REALTIME")
        print(f"   → Giấy phẳng, chỉ nghiêng nhẹ")
        print(f"   → Perspective transform đủ (< 1 giây)")
    
    elif severity == 'MODERATE':
        print(f"\nCHẤT LƯỢNG VỪA PHẢI => TPS CẦN 15-25 GIÂY")
        print(f"   => Giấy có độ cong nhẹ hoặc gấp nếp")
        print(f"   => TPS sẽ sửa nhưng MẤT THỜI GIAN")
        print(f"   => Nên chụp lại ảnh phẳng để xử lý nhanh")
        
        if analysis['problem_areas']:
            print(f"\n   Vùng có vấn đề:")
            for pm in analysis['problem_areas'][:3]:
                print(f"   • Marker {pm['id']}: lệch {pm['error']:.1f}px")
    
    else:  # SEVERE
        print(f"\nCHẤT LƯỢNG KÉM => TPS CẦN 30-40+ GIÂY")
        print(f"   => Giấy bị cong/gấp nếp nghiêm trọng")
        print(f"   => TPS sẽ rất chậm")
        print(f"   => KHUYẾN CÁO: CHỤP LẠI ảnh phẳng")
        
        if analysis['problem_areas']:
            print(f"\n   Vùng có vấn đề nghiêm trọng:")
            for pm in analysis['problem_areas']:
                print(f"   • Marker {pm['id']}: lệch {pm['error']:.1f}px")
    
    print("\n" + "="*70)
