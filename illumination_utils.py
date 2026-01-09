"""
Illumination correction utilities - Cân bằng sáng cho giấy có độ sáng không đều
"""
import cv2 as cv
import numpy as np


def normalize_illumination_morphology(img, kernel_size=25):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    
    # Trừ background
    imgNormalized = cv.subtract(img, background)
    imgNormalized = cv.add(imgNormalized, 128)
    
    return imgNormalized


def normalize_illumination_gaussian(img, sigma=50):
    # Tính kernel size từ sigma
    kernel_size = int(6 * sigma) | 1  # Đảm bảo lẻ
    kernel_size = min(kernel_size, 255)  # Giới hạn max
    
    # Blur để ước lượng background
    background = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Trừ background
    imgNormalized = cv.subtract(img, background)
    imgNormalized = cv.add(imgNormalized, 128)
    
    return imgNormalized


def normalize_illumination_division(img):
    img_float = img.astype(np.float32)
    
    # Ước lượng background bằng blur mạnh
    background = cv.GaussianBlur(img, (51, 51), 0).astype(np.float32)
    
    # Tránh chia cho 0
    background = np.maximum(background, 1.0)
    
    # Chia để normalize
    normalized = (img_float / background) * 128.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def adaptive_illumination_correction(img, method='auto'):
    if method == 'auto':
        # Phân tích độ biến thiên sáng
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        # Nếu độ biến thiên cao, dùng division
        if std_val > 40:
            print("[ILLUM] Độ biến thiên cao, dùng division method")
            return normalize_illumination_division(img)
        # Nếu trung bình, dùng morphology
        else:
            print("[ILLUM] Dùng morphology method")
            return normalize_illumination_morphology(img, kernel_size=25)
    
    elif method == 'morphology':
        return normalize_illumination_morphology(img, kernel_size=25)
    elif method == 'gaussian':
        return normalize_illumination_gaussian(img, sigma=50)
    elif method == 'division':
        return normalize_illumination_division(img)
    else:
        raise ValueError(f"Unknown method: {method}")


def enhance_local_contrast(img, clip_limit=2.0, tile_size=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def multi_scale_illumination_correction(img):
    # Scale 1: Xử lý gradient rộng (toàn cục)
    kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE, (51, 51))
    bg_large = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_large)
    img_norm1 = cv.subtract(img, bg_large)
    img_norm1 = cv.add(img_norm1, 128)
    
    # Scale 2: Xử lý bóng trung bình
    kernel_medium = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    bg_medium = cv.morphologyEx(img_norm1, cv.MORPH_OPEN, kernel_medium)
    img_norm2 = cv.subtract(img_norm1, bg_medium)
    img_norm2 = cv.add(img_norm2, 128)
    
    # CLAHE để tăng cường contrast cục bộ
    img_final = enhance_local_contrast(img_norm2, clip_limit=2.0, tile_size=(10, 10))
    
    return img_final


def compare_illumination_methods(img):
    from matplotlib import pyplot as plt
    
    methods = {
        'Original': img,
        'Morphology': normalize_illumination_morphology(img),
        'Gaussian': normalize_illumination_gaussian(img),
        'Division': normalize_illumination_division(img),
        'Multi-scale': multi_scale_illumination_correction(img)
    }
    
    plt.figure(figsize=(15, 10))
    for i, (name, result) in enumerate(methods.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}\nMean={np.mean(result):.1f}, Std={np.std(result):.1f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_steps/Illumination_Comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return methods
