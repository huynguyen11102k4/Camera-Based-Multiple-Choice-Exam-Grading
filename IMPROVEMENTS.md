# CẢI TIẾN XỬ LÝ GIẤY BỊ NẾP GẤP VÀ CONG VÊNH

## Tổng Quan
Hệ thống đã được cải thiện để xử lý hiệu quả các trường hợp giấy bị:
- Nếp gấp mạnh (creases)
- Cong vênh (warping)  
- Nhàu nát, không phẳng
- Bóng do nếp gấp

## Các Cải Tiến Chính

### 1. Tiền Xử Lý Ảnh Nâng Cao (`pipeline.py`)

#### 1.0 Illumination Correction (MỚI) 🌟
- **Mục đích**: Cân bằng sáng toàn cục, loại bỏ gradient sáng do ánh sáng không đều
- **Phương pháp**: Morphological opening với kernel lớn (25x25)
- **Lợi ích**: 
  - Loại bỏ vùng sáng/tối không đều do nếp gấp
  - Giúp adaptive threshold hoạt động tốt hơn
  - Cải thiện đáng kể chất lượng với giấy có bóng

```python
# Ước lượng background bằng morphological opening
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
background = cv.morphologyEx(imgGray, cv.MORPH_OPEN, kernel)

# Trừ background để normalize
imgNormalized = cv.subtract(imgGray, background)
imgNormalized = cv.add(imgNormalized, 128)
```

#### 1.1 Bilateral Filtering
- **Mục đích**: Giảm nhiễu nhưng giữ nguyên cạnh của markers
- **Tham số**: 
  - `d=9`: Đường kính vùng lọc
  - `sigmaColor=75`, `sigmaSpace=75`: Độ mượt màu và không gian
- **Lợi ích**: Giúp phát hiện markers tốt hơn trên giấy nhàu

#### 1.2 CLAHE Kép (Contrast Limited Adaptive Histogram Equalization)
```python
# CLAHE lần 1: Tile lớn cho cân bằng chung
clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))

# CLAHE lần 2: Tile nhỏ cho chi tiết cục bộ
clahe2 = cv.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
```
- **Cải tiến**: Giảm clipLimit và điều chỉnh tile size để tránh over-enhancement
- **Lợi ích**: Cân bằng sáng tốt hơn, tránh tạo noise

#### 1.2.5 Illumination Correction Lần 2 (Sau Warp) 🌟
```python
# Cân bằng sáng lại sau khi warp
kernel_illum = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
bg_warped = cv.morphologyEx(imgWarped, cv.MORPH_OPEN, kernel_illum)
imgWarped_norm = cv.subtract(imgWarped, bg_warped)

# CLAHE nhẹ để tăng contrast cục bộ
clahe_final = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
imgCLAHE_final = clahe_final.apply(imgBilateral2)
```
- **Tại sao**: Warp perspective có thể tạo artifacts và thay đổi phân bố sáng
- **Lợi ích**: Đảm bảo sáng đều trước adaptive threshold

#### 1.3 Morphological Operations
```python
# Close: Lấp các khe nhỏ do nếp gấp
cv.morphologyEx(imgThresh, cv.MORPH_CLOSE, kernel, iterations=1)

# Open: Loại bỏ nhiễu nhỏ
cv.morphologyEx(imgThresh, cv.MORPH_OPEN, kernel, iterations=1)
```


### 3. TPS Warping Thông Minh (`tps_utils.py`)

#### 3.1 Smooth Parameter Tự Động
Hệ thống tự động điều chỉnh độ mượt dựa trên mức độ biến dạng:

```python
# Giấy biến dạng NẶNG (SEVERE)
smooth_param = 1.5  # Smooth cao để xử lý nếp gấp

# Giấy biến dạng VỪA (MODERATE)  
smooth_param = 0.8  # Cân bằng

# Giấy TỐT (GOOD)
smooth_param = 0.3  # Giữ chi tiết
```

#### 3.2 Nội Suy CUBIC
- **Trước**: `INTER_LINEAR`
- **Sau**: `INTER_CUBIC` với `BORDER_REPLICATE`
- **Lợi ích**: Chất lượng ảnh tốt hơn, giảm artifacts

### 4. Adaptive Thresholding Cải Tiến

#### 4.1 Bilateral Filter Trước Threshold
```python
imgBilateral2 = cv.bilateralFilter(imgWarped, d=5, sigmaColor=50, sigmaSpace=50)
```
- Giảm nhiễu từ nếp gấp trước khi threshold

#### 4.2 Block Size Lớn Hơn
```python
# Trước: blockSize=201 hoặc 251
# Sau: blockSize=201, C=8
cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv.THRESH_BINARY, 201, 8)
```
- **Cải tiến**: Giảm blockSize nhưng tăng C parameter
- **Lý do**: BlockSize quá lớn (251) có thể tạo artifacts, C=8 bù trừ bằng threshold cao hơn
- Xử lý tốt hơn bóng từ nếp gấp

### 4.5 Module Illumination Utils (MỚI) 🌟

File `illumination_utils.py` cung cấp nhiều phương pháp cân bằng sáng:

#### Các Phương Pháp Có Sẵn:
1. **`normalize_illumination_morphology()`**: Dùng morphological opening
2. **`normalize_illumination_gaussian()`**: Dùng Gaussian blur  
3. **`normalize_illumination_division()`**: Chia cho background - tốt cho gradient mạnh
4. **`multi_scale_illumination_correction()`**: Đa tỷ lệ - tốt nhất cho trường hợp phức tạp
5. **`adaptive_illumination_correction()`**: Tự động chọn phương pháp

#### Demo So Sánh:
```bash
python demo_illumination.py
```
Xem `debug_steps/Illumination_Comparison.png` để so sánh các phương pháp.

### 5. Mesh-Based Dewarping (Module Mới)

File `mesh_dewarp.py` cung cấp các hàm để xử lý biến dạng cục bộ:

#### 5.1 `create_mesh_grid()`
Chia ảnh thành lưới các ô nhỏ để warp riêng

#### 5.2 `mesh_based_warp()`
Warp dựa trên lưới với interpolation RBF

#### 5.3 `adaptive_local_warp()`
Warp từng vùng nhỏ - tốt cho nếp gấp cục bộ

**⚠️ LÀM SAO SỬ DỤNG:**

**Tùy chọn A - Chạy độc lập:**
```bash
python demo_mesh_dewarp.py
```
Script này sẽ áp dụng mesh dewarping và lưu kết quả vào `debug_steps/`.

**Tùy chọn B - Tích hợp vào pipeline chính:**
Module này **chưa được tích hợp mặc định** vào [pipeline.py](pipeline.py) vì:
- TPS warping đã đủ tốt cho hầu hết trường hợp
- Mesh dewarping chậm hơn và phức tạp hơn
- Chỉ cần thiết cho giấy bị nếp gấp **CỰC KỲ MẠNH**

Để tích hợp, thêm vào [pipeline.py](pipeline.py) sau TPS warp:
```python
from mesh_dewarp import mesh_based_warp

# Sau dòng TPS warp, thêm:
if analysis['severity'] == 'SEVERE' and analysis['max_error'] > 50:
    print("[MESH] Áp dụng mesh dewarping cho biến dạng cực mạnh...")
    imgWarped = mesh_based_warp(imgWarped, new_marker_dict, MARKER_CENTERS, grid_size=150)
```

## Cách Sử Dụng

### Chạy Pipeline
```bash
python demo.py images/img2.png
```

### Kiểm Tra Kết Quả
Xem các file debug trong thư mục `debug_steps/`:
- `01a_Illumination_Normalized.png`: Sau cân bằng sáng toàn cục
- `01b_Bilateral_Filter.png`: Sau bilateral filter
- `02_Enhanced_CLAHE.png`: Sau CLAHE kép
- `03_TPS_Dewarped.png`: Sau TPS warping
- `04a_Rebalanced_After_Warp.png`: Cân bằng sáng lần 2
- `04b_Bilateral_Before_Threshold.png`: Trước threshold
- `04c_CLAHE_Final.png`: CLAHE cuối cùng
- `05_Adaptive_Threshold.png`: Sau adaptive threshold
- `06_Final_Thresh.png`: Kết quả cuối

## Phân Tích Tự Động

Hệ thống tự động phân tích chất lượng giấy:

```
📊 PHÂN TÍCH CHẤT LƯỢNG GIẤY
================================================
📏 Độ lệch markers sau perspective transform:
   • Lệch tối đa: 45.2 pixels
   • Lệch trung bình: 28.1 pixels
   • Số markers kiểm tra: 17

❌ CHẤT LƯỢNG KÉM
   → Giấy bị nếp gấp mạnh hoặc cong nhiều
   → TPS với smooth cao sẽ được áp dụng
```

## Tham Số Có Thể Điều Chỉnh

### Để Xử Lý Giấy Rất Nhàu
Trong `pipeline.py`:
```python
# Tăng illumination correction
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (51, 51))  # Lớn hơn

# Tăng bilateral filter
imgBilateral = cv.bilateralFilter(imgGray, d=11, sigmaColor=100, sigmaSpace=100)

# Tăng CLAHE
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
```

### Để Xử Lý Giấy Có Gradient Sáng Mạnh
```python
# Dùng division method thay vì morphology
from illumination_utils import normalize_illumination_division
imgNormalized = normalize_illumination_division(imgGray)

# Hoặc dùng multi-scale
from illumination_utils import multi_scale_illumination_correction
imgNormalized = multi_scale_illumination_correction(imgGray)
```

### Để Giảm Xử Lý (Giấy Tốt)
```python
# Giảm bilateral filter
imgBilateral = cv.bilateralFilter(imgGray, d=5, sigmaColor=50, sigmaSpace=50)

# Giảm CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
```

### Điều Chỉnh TPS Smooth
Trong `smart_decision.py`:
```python
if analysis['severity'] == 'SEVERE':
    smooth_param = 2.0  # Tăng để xử lý mạnh hơn
```

## Lưu Ý Quan Trọng

1. **Bilateral Filter**: Chậm hơn Gaussian nhưng giữ cạnh tốt hơn
2. **CLAHE Kép**: Cần thiết cho giấy có độ sáng không đều
3. **Morphological Ops**: Không nên iterations > 2, sẽ làm mất chi tiết
4. **TPS Smooth**: Giá trị quá cao (>2.5) sẽ làm mất độ chính xác

## So Sánh Trước/Sau

| Thành Phần | Trước | Sau |
|------------|-------|-----|
| Illumination Correction | ❌ Không có | ✅ 2 lần (pre + post warp) |
| Bilateral Filter | ❌ Không có | ✅ 2 lần (pre + post) |
| CLAHE | 1 lần | ✅ 3 lần với tham số khác nhau |
| ArUco Detection | 1 lần | ✅ Multi-pass (3 lần) |
| Corner Refinement | ❌ Mặc định | ✅ SUBPIX |
| TPS Smooth | 0.0 (fixed) | ✅ 0.3-1.5 (adaptive) |
| Interpolation | LINEAR | ✅ CUBIC |
| Morphological | ❌ Không có | ✅ Close + Open |
| Adaptive Block | 201 | ✅ 201 + C=8 |
| Mesh Dewarping | ❌ Không có | ✅ Module riêng |
| Illumination Utils | ❌ Không có | ✅ 5 phương pháp |

## Kết Luận

Các cải tiến này giúp hệ thống xử lý tốt các trường hợp:
- ✅ Giấy bị nếp gấp ngang/dọc
- ✅ Giấy cong vênh không đều
- ✅ Bóng do nếp gấp
- ✅ Markers bị che khuất một phần
- ✅ Độ sáng không đều

Để có kết quả tốt nhất, đảm bảo:
- Ảnh chụp có độ phân giải đủ cao (>= 2000px width)
- Đủ ánh sáng, tránh bóng đen quá đậm
- Markers không bị che khuất hoàn toàn

---

## 🔧 Hướng Dẫn Sử Dụng Mesh Dewarping

### Khi Nào Dùng Mesh Dewarping?

✅ **NÊN dùng** khi:
- Giấy có nếp gấp CỰC KỲ MẠNH (gấp đôi, gấp nhiều lần)
- TPS warping không đủ tốt (max_error > 50 pixels)
- Có vùng biến dạng cục bộ rất lớn

❌ **KHÔNG cần** khi:
- Giấy chỉ cong nhẹ hoặc nghiêng
- TPS đã cho kết quả tốt (max_error < 30 pixels)  
- Cần xử lý nhanh

### Cách Sử Dụng

#### Option 1: Chạy Demo Độc Lập
```bash
python demo_mesh_dewarp.py
```
Xem kết quả trong `debug_steps/Mesh_*.png`

#### Option 2: Tích Hợp Vào Pipeline

Thêm vào [pipeline.py](pipeline.py) sau dòng TPS warp (khoảng line 90):

```python
# Import ở đầu file
from mesh_dewarp import mesh_based_warp

# Sau khi TPS warp xong
if analysis['severity'] == 'SEVERE' and analysis['max_error'] > 50:
    print("[MESH] Áp dụng mesh dewarping bổ sung...")
    imgWarped = mesh_based_warp(
        imgWarped, 
        new_marker_dict, 
        MARKER_CENTERS, 
        grid_size=150
    )
    save_step(imgWarped, "03b_Mesh_Dewarped")
```

### Điều Chỉnh Tham Số

```python
# grid_size: Kích thước ô lưới
mesh_based_warp(img, markers, expected, grid_size=100)  # Chi tiết hơn, chậm hơn
mesh_based_warp(img, markers, expected, grid_size=200)  # Nhanh hơn, ít chi tiết
```

### So Sánh: TPS vs Mesh

| Đặc Điểm | TPS Warp | Mesh Dewarp |
|----------|----------|-------------|
| Tốc độ | ⚡ Nhanh | 🐢 Chậm hơn 2-3x |
| Chất lượng | ✅ Tốt cho hầu hết TH | ✅✅ Tốt hơn cho nếp gấp mạnh |
| Phức tạp | Đơn giản | Phức tạp hơn |
| Sử dụng | Mặc định | Bổ sung khi cần |
| Markers cần | >= 3 | >= 5 để hiệu quả |
