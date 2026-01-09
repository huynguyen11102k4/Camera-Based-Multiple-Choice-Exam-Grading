"""
Demo so sánh các phương pháp cân bằng sáng
"""
import os
import cv2 as cv
from illumination_utils import compare_illumination_methods

if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/img2.png')
    
    print("=" * 70)
    print("SO SÁNH CÁC PHƯƠNG PHÁP CÂN BẰNG SÁNG")
    print("=" * 70)
    
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    print("\nĐang so sánh các phương pháp...")
    print("- Original")
    print("- Morphology (Opening)")
    print("- Gaussian Blur")
    print("- Division")
    print("- Multi-scale")
    
    results = compare_illumination_methods(img)
    
    print("\n✅ Kết quả đã lưu vào debug_steps/Illumination_Comparison.png")
    print("\nGợi ý:")
    print("- Nếu có gradient sáng nhẹ → Morphology")
    print("- Nếu có bóng mạnh → Division")
    print("- Nếu có cả hai → Multi-scale")
