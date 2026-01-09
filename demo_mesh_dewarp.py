import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from config import MARKER_CENTERS, FULL_WIDTH, FULL_HEIGHT
from marker_utils import findArucoMarkers, getMainCorners
from warp_utils import warpRegion
from mesh_dewarp import mesh_based_warp, adaptive_local_warp
from debug_utils import save_step


def demo_mesh_pipeline(imgPath):
    """Pipeline với mesh-based dewarping"""
    
    img = cv.imread(imgPath)
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    print("=" * 70)
    print("DEMO: MESH-BASED DEWARPING")
    print("=" * 70)
    
    # B1: CLAHE
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgGray)
    
    # B2: Detect markers
    try:
        marker_dict = findArucoMarkers(imgCLAHE)
    except RuntimeError as e:
        print("[LỖI MARKER]", e)
        return
    
    # B3: Warp toàn bộ ảnh với 4 góc chính
    main_corners = getMainCorners(marker_dict)
    imgWarped, M_main = warpRegion(imgCLAHE, main_corners, FULL_WIDTH, FULL_HEIGHT)
    save_step(imgWarped, "Mesh_01_Perspective_Warped")
    
    # B4: Detect markers trên ảnh đã warp
    new_marker_dict = findArucoMarkers(imgWarped)
    
    # B5: Áp dụng MESH-BASED WARP
    print("\n[MESH DEWARP] Áp dụng mesh-based warping...")
    imgMeshWarped = mesh_based_warp(
        imgWarped, 
        new_marker_dict, 
        MARKER_CENTERS, 
        grid_size=150
    )
    save_step(imgMeshWarped, "Mesh_02_Mesh_Dewarped")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(imgGray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("After Perspective")
    plt.imshow(imgWarped, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("After Mesh Dewarp")
    plt.imshow(imgMeshWarped, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_steps/Mesh_Comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nHoàn tất! Kiểm tra kết quả trong debug_steps/")
    print("   - Mesh_01_Perspective_Warped.png")
    print("   - Mesh_02_Mesh_Dewarped.png")
    print("   - Mesh_Comparison.png")


if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/img2.png')
    demo_mesh_pipeline(imgPath)
