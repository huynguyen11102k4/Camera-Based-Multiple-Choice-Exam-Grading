# core.py
from __future__ import annotations

import logging
import os
from typing import Tuple

import cv2 as cv
import numpy as np

# ========== Constants ==========
# Tôi gom constant vào một chỗ để bạn đổi kích thước A4, tên file... dễ hơn.
A4_PX: Tuple[int, int] = (2480, 3508)  # 300 DPI portrait (W, H)
TEMPLATE_LAYOUT_FILE = "samples/template_marker_layout.json"
ROI_CALIB_FILE = "samples/roi.json"
DEBUG_DIR = "debug_steps"

TAG_SYSTEM = "apriltag"  # "aruco" / "apriltag"


# ArUco dictionary
ARUCO_DICT = cv.aruco.DICT_4X4_50

# AprilTag family
APRILTAG_FAMILY = cv.aruco.DICT_APRILTAG_16h5

# Độ dày vòng tròn vẽ kết quả để nhìn rõ trên ảnh scan.
CIRCLE_STROKE = 4

# ========== Logging ==========
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omr")


# ========== File / IO Utils ==========
def safe_mkdir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại.

    Tôi tách ra hàm riêng vì thao tác này dùng ở nhiều nơi (debug, output, ...).
    """
    os.makedirs(path, exist_ok=True)


def safe_cleanup_dir(path: str) -> None:
    """Xoá toàn bộ file trong thư mục (không xoá thư mục).

    Tôi làm như vậy để debug_dir luôn sạch cho mỗi lần chạy pipeline.
    """
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass


# ========== Display Helper (zoom/pan) ==========
def show_and_save_step(
    idx: int,
    name: str,
    img: np.ndarray,
    folder: str,
    interactive: bool = True,
) -> None:
    """Lưu ảnh từng bước và *tuỳ chọn* hiển thị với chức năng zoom/pan.

    Tôi giữ `interactive` để bạn dễ disable UI khi chạy batch/CI, vì khi chạy trên server thường không có màn hình.
    """
    safe_mkdir(folder)
    path = os.path.join(folder, f"{idx:02d}_{name}.png")
    ok = cv.imwrite(path, img)
    log.debug(
        f"[Step {idx}] Saved step '{name}' -> {path}, "
        f"ok={ok}, shape={img.shape}"
    )

    if not interactive:
        # Khi chạy không có UI, ta chỉ lưu file mà không mở cửa sổ vì tránh lỗi trên môi trường headless.
        return

    window = f"{idx:02d}_{name}"
    h, w = img.shape[:2]
    screen_w, screen_h = 1920, 1080
    scale = min(screen_w / w, screen_h / h, 1.0)
    base_display = cv.resize(img, (int(w * scale), int(h * scale)))
    win_w = int(min(screen_w, base_display.shape[1]) * 0.9)
    win_h = int(min(screen_h, base_display.shape[0]) * 0.9)
    log.debug(
        f"[Step {idx}] window='{window}', "
        f"base_display={base_display.shape}, win=({win_w},{win_h}), "
        f"scale={scale:.3f}"
    )

    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.moveWindow(window, 800, 0)
    cv.resizeWindow(window, win_w, win_h)

    # Trạng thái xem
    zoom = 0.9
    pan_x = 0.0
    pan_y = 0.0
    dragging = False
    start_x = start_y = 0

    def update():
        """Cập nhật vùng hiển thị theo zoom/pan, tự pad nếu thiếu.

        Tôi gom vào hàm riêng để callback chuột gọn hơn, vì vậy dễ bảo trì.
        """
        nonlocal zoom, pan_x, pan_y
        zoom = float(np.clip(zoom, 0.1, 10.0))
        view_w, view_h = win_w, win_h
        zoomed = cv.resize(base_display, None, fx=zoom, fy=zoom)
        zh, zw = zoomed.shape[:2]
        pan_x = float(np.clip(pan_x, 0, max(0, zw - view_w)))
        pan_y = float(np.clip(pan_y, 0, max(0, zh - view_h)))
        x0, y0 = int(pan_x), int(pan_y)
        x1, y1 = min(x0 + view_w, zw), min(y0 + view_h, zh)
        cropped = zoomed[y0:y1, x0:x1]
        pad_h = max(0, view_h - cropped.shape[0])
        pad_w = max(0, view_w - cropped.shape[1])
        if pad_h or pad_w:
            cropped = cv.copyMakeBorder(
                cropped,
                0,
                pad_h,
                0,
                pad_w,
                cv.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        cv.imshow(window, cropped)

    def on_mouse(event, x, y, flags, _param):
        """Chuột trái kéo để pan, lăn để zoom.

        Tôi bind sự kiện chuột để bạn có trải nghiệm inspect ảnh thuận tiện hơn.
        """
        nonlocal dragging, start_x, start_y, pan_x, pan_y, zoom
        if event == cv.EVENT_LBUTTONDOWN:
            dragging = True
            start_x, start_y = x, y
        elif event == cv.EVENT_MOUSEMOVE and dragging:
            dx, dy = x - start_x, y - start_y
            pan_x -= dx
            pan_y -= dy
            start_x, start_y = x, y
            update()
        elif event == cv.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv.EVENT_MOUSEWHEEL:
            zoom *= 1.1 if flags > 0 else 1 / 1.1
            update()

    cv.setMouseCallback(window, on_mouse)
    log.info(f"[Step {idx}] {name} — Enter=next, Esc=quit")
    update()
    while True:
        key = cv.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # Esc
            cv.destroyAllWindows()
            raise SystemExit("Stopped by user.")
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Self-test rất nhẹ: tạo thư mục debug_core để bạn biết core hoạt động, vì vậy không phụ thuộc ảnh hay OpenCV UI.
    print("[core] Self-test: creating debug_core directory...")
    safe_mkdir("debug_core")
    print("Created folder: debug_core (or already existed).")
    print("core.py self-test complete.")
