import cv2 as cv
import numpy as np


def detect_crease_regions(img_gray, sensitivity=1.0):
    h, w = img_gray.shape
    
    grad_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
    
    threshold_val = max(20, int(50 / sensitivity))
    _, crease_edges = cv.threshold(gradient_mag, threshold_val, 255, cv.THRESH_BINARY)
    
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    crease_dilated = cv.dilate(crease_edges, kernel_dilate, iterations=2)
    
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    crease_mask = cv.morphologyEx(crease_dilated, cv.MORPH_CLOSE, kernel_close, iterations=1)
    
    contours, _ = cv.findContours(crease_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    min_area = (h * w) * 0.001
    crease_regions = []
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area:
            x, y, w_box, h_box = cv.boundingRect(cnt)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_box = min(w - x, w_box + 2 * padding)
            h_box = min(h - y, h_box + 2 * padding)
            crease_regions.append((x, y, w_box, h_box))
    
    return crease_mask, crease_regions


def enhance_crease_region(img_gray, region_bbox, strength=1.5):
    x, y, w, h = region_bbox
    result = img_gray.copy()
    
    region = img_gray[y:y+h, x:x+w].copy()
    
    d_val = int(9 + strength * 2)
    sigma_color = int(75 + strength * 25)
    sigma_space = int(75 + strength * 25)
    
    region_smoothed = cv.bilateralFilter(region, d=d_val, 
                                          sigmaColor=sigma_color, 
                                          sigmaSpace=sigma_space)
    grad_x = cv.Sobel(region, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(region, cv.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient_norm = np.uint8(gradient / gradient.max() * 255)
    
    _, inpaint_mask = cv.threshold(gradient_norm, 60, 255, cv.THRESH_BINARY)
    
    kernel_inpaint = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv.dilate(inpaint_mask, kernel_inpaint, iterations=1)
    
    if np.sum(inpaint_mask) > 0:
        region_inpainted = cv.inpaint(region_smoothed, inpaint_mask, 
                                      inpaintRadius=int(3 * strength), 
                                      flags=cv.INPAINT_TELEA)
    else:
        region_inpainted = region_smoothed
    
    alpha = 0.7 
    region_blended = cv.addWeighted(region_inpainted, alpha, region, 1 - alpha, 0)
    
    result[y:y+h, x:x+w] = region_blended
    
    return result


def process_creases(img_gray, sensitivity=1.0, strength=1.5, debug=False):
    crease_mask, crease_regions = detect_crease_regions(img_gray, sensitivity)
    
    if debug:
        print(f"[CREASE] Phát hiện {len(crease_regions)} vùng nếp gấp")
        for i, (x, y, w, h) in enumerate(crease_regions):
            print(f"  Region {i+1}: x={x}, y={y}, w={w}, h={h}")
    
    img_enhanced = img_gray.copy()
    for i, bbox in enumerate(crease_regions):
        if debug:
            print(f"[CREASE] Xử lý region {i+1}/{len(crease_regions)}...")
        img_enhanced = enhance_crease_region(img_enhanced, bbox, strength)
    
    return img_enhanced, crease_mask


def adaptive_sharpen_after_warp(img_gray, strength=1.3):
    blurred = cv.GaussianBlur(img_gray, (0, 0), sigmaX=3, sigmaY=3)
    sharpened = cv.addWeighted(img_gray, 1.0 + strength, blurred, -strength, 0)
    
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened
