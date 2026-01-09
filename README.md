# 🚀 Quick Start - Smart OMR System

## Chạy nhanh:

### **Cách 1: Interactive Mode (Khuyến nghị cho test)**
```bash
python demo.py images/img2.png
```
Hệ thống sẽ phân tích và **hỏi bạn** nếu cần xử lý chậm hơn.

### **Cách 2: Auto Mode (Cho production)**
```bash
python demo.py images/img2.png --auto
```
Hệ thống tự động quyết định, không hỏi.

---

## 📊 Hệ thống sẽ báo cáo:

### ✅ **Giấy TỐT** (lệch < 15px)
```
📊 PHÂN TÍCH CHẤT LƯỢNG GIẤY
✅ CHẤT LƯỢNG TỐT
   → Giấy phẳng, chỉ nghiêng nhẹ
   → Perspective Transform đủ chính xác
   → Thời gian xử lý: ~0.5 giây
```
→ Tự động dùng Perspective, không hỏi

### ⚠️ **Giấy VỪA** (lệch 15-40px)
```
⚠️  CHẤT LƯỢNG VỪA PHẢI
   → Giấy có độ cong nhẹ
   → Nên dùng refinement (~1-2 giây)
   
Bạn có muốn dùng refinement? (y/n):
```
→ Hỏi user chọn

### ❌ **Giấy KÉM** (lệch > 40px)
```
❌ CHẤT LƯỢNG KÉM
   → Giấy bị cong/gấp nếp nghiêm trọng
   → CẦN refinement (~30-40 giây)
   
Bạn có muốn dùng refinement? (y/n):
```
→ Khuyến nghị mạnh refine hoặc scan lại

---

## 🎯 Khi nào chọn gì?

| User chọn | Tốc độ | Độ chính xác | Khi nào dùng |
|-----------|--------|--------------|--------------|
| **[n] No** | 0.5s | 85-90% | Giấy tốt, test nhanh |
| **[y] Yes** | 1-40s | 95-98% | Giấy cong, cần chính xác |

---

## 📝 Chi tiết: [SOLUTION.md](SOLUTION.md)
