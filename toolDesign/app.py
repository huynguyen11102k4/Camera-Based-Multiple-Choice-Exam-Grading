# app.py
import streamlit as st
import cv2 as cv
import numpy as np
import json
import os
from PIL import Image
import io
import base64
from datetime import datetime

# === TỰ ĐỘNG TẠO THƯ MỤC ===
os.makedirs("templates", exist_ok=True)
os.makedirs("output", exist_ok=True)

# === CẤU HÌNH CHUNG ===
st.set_page_config(page_title="Quiz Designer Pro MAX", layout="wide")
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
MARKER_SIZE = 150
# Lưu ý: 1575x2316 ~ 200dpi A4. Nếu muốn 300dpi chuẩn A4: dùng 2480x3508
CANVAS_W, CANVAS_H = 1575, 2316

# === SESSION STATE MẶC ĐỊNH ===
defaults = {
    'markers': [],         # list of tuples: (id, x, y, label). x < 0 nghĩa là ẩn tạm (chưa đặt)
    'regions': [],         # list of dict: {region:[x0,x1,y0,y1], start_question, num_questions, options_per_question, title, rois:[[x0,x1,y0,y1,qidx,optIdx], ...]}
    'bg_image': None,      # ảnh nền đã căn giữa theo canvas
    'canvas': None,        # cache canvas đã vẽ
    'template_name': 'mau_trac_nghiem',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# === TIỆN ÍCH ARUCO + DÁN ẢNH AN TOÀN BIÊN ===
def _make_marker_img(mid, size):
    """Sinh ảnh marker ArUco. Hỗ trợ cả API cũ (drawMarker) và mới (generateImageMarker)."""
    try:
        marker = cv.aruco.generateImageMarker(ARUCO_DICT, int(mid), int(size))
    except Exception:
        marker = np.zeros((size, size), dtype=np.uint8)
        cv.aruco.drawMarker(ARUCO_DICT, int(mid), int(size), marker, 1)
    return cv.cvtColor(marker, cv.COLOR_GRAY2BGR)

def _paste_img_safe(dst, src, center_xy):
    """Dán src vào dst tại tâm center_xy, tự cắt biên nếu tràn."""
    h, w = dst.shape[:2]
    H, W = src.shape[:2]
    cx, cy = center_xy
    x0 = cx - W // 2
    y0 = cy - H // 2
    x1 = x0 + W
    y1 = y0 + H

    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)

    sx1 = min(W, W - (x1 - w) if x1 > w else W)
    sy1 = min(H, H - (y1 - h) if y1 > h else H)

    if sx0 < sx1 and sy0 < sy1:
        dst[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = src[sy0:sy1, sx0:sx1]

# === HÀM VẼ CANVAS ===
def draw_canvas(bg_img=None):
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
    if bg_img is not None:
        # bg_img được kỳ vọng đã cùng kích thước canvas
        canvas = bg_img.copy()

    # Lưới (100px)
    for x in range(0, CANVAS_W, 100):
        cv.line(canvas, (x, 0), (x, CANVAS_H), (240, 240, 240), 1)
    for y in range(0, CANVAS_H, 100):
        cv.line(canvas, (0, y), (CANVAS_W, y), (240, 240, 240), 1)

    # Vẽ ArUco markers an toàn biên
    for mid, x, y, label in st.session_state.markers:
        if x < 0:
            continue  # ẩn tạm (chưa đặt)
        marker_bgr = _make_marker_img(mid, MARKER_SIZE)
        _paste_img_safe(canvas, marker_bgr, (x, y))
        tl = (x - MARKER_SIZE//2, y - MARKER_SIZE//2)
        cv.putText(canvas, f"ID:{mid}", (max(0, tl[0]), max(20, tl[1]-10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if label:
            cv.putText(canvas, label, (max(0, x-50), max(20, y - MARKER_SIZE//2 - 20)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    # Vẽ các vùng
    for r in st.session_state.regions:
        x0, x1, y0, y1 = r['region']  # format thống nhất [x0,x1,y0,y1]
        cv.rectangle(canvas, (x0, y0), (x1, y1), (0, 150, 0), 2)
        cv.putText(canvas, r.get('title', ''), (x0, max(20, y0 - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

    return canvas

# === LƯU MẪU ===
def save_template(name):
    folder = f"templates/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder, exist_ok=True)

    canvas = draw_canvas(st.session_state.bg_image)
    cv.imwrite(f"{folder}/template.png", canvas)

    layout = {
        "image_size": [CANVAS_W, CANVAS_H],
        "aruco_markers": [
            {"id": int(m[0]), "center": [int(m[1]), int(m[2])], "size": int(MARKER_SIZE), "label": m[3]}
            for m in st.session_state.markers if m[1] >= 0
        ],
        "answer_regions": st.session_state.regions
    }
    with open(f"{folder}/layout.json", 'w', encoding='utf-8') as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    st.success(f"Đã lưu mẫu: `{folder}`")

# === LOAD MẪU ===
def load_template(folder):
    img_path = os.path.join("templates", folder, "template.png")
    layout_path = os.path.join("templates", folder, "layout.json")

    # Load layout trước để chủ động tái vẽ (tránh "chồng" marker từ ảnh sẵn có)
    if os.path.exists(layout_path):
        with open(layout_path, 'r', encoding='utf-8') as f:
            layout = json.load(f)
        st.session_state.markers = [
            (int(m["id"]), int(m["center"][0]), int(m["center"][1]), m.get("label", ""))
            for m in layout.get("aruco_markers", [])
        ]
        st.session_state.regions = layout.get("answer_regions", [])

    # Ảnh nền: để trắng rồi vẽ lại theo layout; không dùng template.png để tránh chồng hình
    st.session_state.bg_image = None
    st.rerun()

# === XUẤT PDF ===
def export_pdf():
    canvas = draw_canvas(st.session_state.bg_image)
    pil_img = Image.fromarray(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    pdf_buffer = io.BytesIO()
    pil_img.save(pdf_buffer, format="PDF", resolution=300.0)
    pdf_bytes = pdf_buffer.getvalue()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="quiz_template.pdf">TẢI PDF (300dpi)</a>'
    st.markdown(href, unsafe_allow_html=True)

# === QUERY PARAMS: HỖ TRỢ API MỚI VÀ CŨ ===
def _get_qp():
    try:
        # Streamlit >= 1.31
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_qp(**kwargs):
    try:
        # API mới: làm sạch và set lại
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

# === SIDEBAR ===
st.sidebar.image("https://img.icons8.com/fluency/48/000000/document.png", width=48)
st.sidebar.title("Quiz Designer Pro MAX")

tab1, tab2, tab3, tab4 = st.sidebar.tabs(["ArUco", "Vùng", "Mẫu", "Xuất"])

# --- TAB 1: ArUco ---
with tab1:
    st.subheader("Click để đặt Marker")
    col1, col2 = st.columns(2)
    new_id = col1.number_input("ID", 0, 49, 10, key="id_input")
    new_label = col2.text_input("Nhãn", "", key="label_input")
    if st.button("Chuẩn bị đặt Marker"):
        st.session_state.pending_marker = (int(new_id), new_label)
        # thêm bản ghi ẩn tạm (x = -100)
        st.session_state.markers.append((int(new_id), -100, -100, new_label))
        st.rerun()

    if st.session_state.markers:
        st.write("**Marker hiện có:**")
        to_remove = []
        for i, (mid, x, y, label) in enumerate(st.session_state.markers):
            if x < 0:
                continue
            cols = st.columns([1, 1, 1, 2, 1])
            cols[0].write(int(mid))
            cols[1].write(int(x))
            cols[2].write(int(y))
            cols[3].write(label)
            if cols[4].button("Xóa", key=f"delm_{i}"):
                to_remove.append(i)
        for i in reversed(to_remove):
            st.session_state.markers.pop(i)
        if to_remove:
            st.rerun()

# --- TAB 2: Vùng câu trả lời ---
with tab2:
    st.subheader("Thêm vùng câu trả lời")
    with st.expander("Cấu hình vùng"):
        c1, c2 = st.columns(2)
        x0 = c1.number_input("X0", 0, CANVAS_W, 300, key="rx0")
        x1 = c2.number_input("X1", 0, CANVAS_W, 800, key="rx1")
        y0 = c1.number_input("Y0", 0, CANVAS_H, 800, key="ry0")
        y1 = c2.number_input("Y1", 0, CANVAS_H, 2200, key="ry1")
        start = c1.number_input("Câu bắt đầu", 1, 1000, 1, key="rstart")
        num = c2.number_input("Số câu", 1, 100, 25, key="rnum")
        opts = c1.number_input("Số phương án/1 câu", 2, 10, 5, key="ropts")
        title = c2.text_input("Tiêu đề", "Câu 1-25", key="rtitle")

    if st.button("Thêm vùng"):
        # Validate đơn giản
        if not (0 <= x0 < x1 <= CANVAS_W and 0 <= y0 < y1 <= CANVAS_H):
            st.error("Toạ độ vùng không hợp lệ (cần 0 ≤ X0 < X1 ≤ W và 0 ≤ Y0 < Y1 ≤ H).")
        elif num <= 0 or opts <= 0:
            st.error("Số câu và số phương án phải > 0.")
        else:
            region = {
                "region": [int(x0), int(x1), int(y0), int(y1)],
                "start_question": int(start),
                "num_questions": int(num),
                "options_per_question": int(opts),
                "title": title,
                "rois": []
            }
            row_h = (y1 - y0) / num
            col_w = (x1 - x0) / opts
            shrink = 0.15  # co vào trong để tránh chạm biên
            for q in range(int(num)):
                yy0 = int(y0 + q * row_h)
                yy1 = int(y0 + (q + 1) * row_h)
                dy = int((yy1 - yy0) * shrink)
                for o in range(int(opts)):
                    xx0 = int(x0 + o * col_w)
                    xx1 = int(x0 + (o + 1) * col_w)
                    dx = int((xx1 - xx0) * shrink)
                    # ROI theo format THỐNG NHẤT [x0,x1,y0,y1,question_index,option_index]
                    roi = [xx0 + dx, xx1 - dx, yy0 + dy, yy1 - dy, int(start) + q, o]
                    region["rois"].append(roi)
            st.session_state.regions.append(region)
            st.rerun()

# --- TAB 3: Quản lý mẫu ---
with tab3:
    st.subheader("Quản lý mẫu")
    template_name = st.text_input("Tên mẫu", st.session_state.template_name)
    st.session_state.template_name = template_name
    if st.button("Lưu mẫu hiện tại"):
        save_template(template_name)

    st.write("**Mẫu đã lưu:**")
    templates = []
    if os.path.exists("templates"):
        templates = [f for f in os.listdir("templates") if os.path.isdir(os.path.join("templates", f))]
    if templates:
        for t in templates:
            if st.button(f"Load: {t}", key=f"load_{t}"):
                load_template(t)
    else:
        st.info("Chưa có mẫu nào.")

# --- TAB 4: Xuất ---
with tab4:
    st.subheader("Xuất file")
    if st.button("Xuất PNG + JSON"):
        canvas = draw_canvas(st.session_state.bg_image)
        cv.imwrite("output/template.png", canvas)
        layout = {
            "image_size": [CANVAS_W, CANVAS_H],
            "aruco_markers": [
                {"id": int(m[0]), "center": [int(m[1]), int(m[2])], "size": int(MARKER_SIZE), "label": m[3]}
                for m in st.session_state.markers if m[1] >= 0
            ],
            "answer_regions": st.session_state.regions
        }
        with open("output/layout.json", 'w', encoding='utf-8') as f:
            json.dump(layout, f, indent=2, ensure_ascii=False)
        st.success("Xuất thành công vào `output/`!")

    if st.button("In PDF (300dpi)"):
        export_pdf()

# === MAIN CANVAS ===
st.title("Quiz Designer Pro MAX")
col1, col2 = st.columns([3, 1])

with col1:
    # Tải ảnh nền và căn vừa canvas, đặt giữa
    uploaded = st.file_uploader("Tải ảnh nền (A4 scan)", type=['png','jpg','jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_cv = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv.resize(img_cv, (new_w, new_h))
        bg = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
        x_offset = (CANVAS_W - new_w) // 2
        y_offset = (CANVAS_H - new_h) // 2
        bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        st.session_state.bg_image = bg
        st.rerun()


    # Hiển thị canvas (đặt container để JS chọn đúng ảnh)
    with st.container():
        st.markdown('<div id="canvas-holder"></div>', unsafe_allow_html=True)
        canvas = draw_canvas(st.session_state.bg_image)
        st.session_state.canvas = canvas
        st.image(canvas, channels="BGR", use_container_width=True)  # <— quan trọng



    # === CLICK ĐỂ ĐẶT MARKER ===
    # === CLICK ĐỂ ĐẶT MARKER ===
    if 'pending_marker' in st.session_state:
        mid, label = st.session_state.pending_marker
        st.info(f"Click trên ảnh để đặt marker **ID {mid}**")
        js = """
        <script>
        (function() {
        function bind() {
            // Lấy tất cả ảnh đang hiển thị (Streamlit có thể bọc nhiều lớp)
            const imgs = Array.from(document.querySelectorAll('div[data-testid="stImage"] img'));
            if (!imgs.length) return false;
            // Chọn ảnh cuối (thường là canvas vừa render)
            const img = imgs[imgs.length - 1];
            if (!img) return false;

            img.style.cursor = 'crosshair';
            img.onclick = function(e) {
            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const scale_x = """ + str(CANVAS_W) + """ / rect.width;
            const scale_y = """ + str(CANVAS_H) + """ / rect.height;
            const real_x = Math.round(x * scale_x);
            const real_y = Math.round(y * scale_y);

            const url = new URL(window.location.href);
            url.searchParams.set('place', real_x + ',' + real_y);
            window.location.href = url.toString(); // trigger rerun
            };
            return true;
        }

        // Thử bind ngay
        if (bind()) return;

        // Nếu ảnh chưa xuất hiện, dùng MutationObserver để đợi DOM xong
        const obs = new MutationObserver(() => {
            if (bind()) obs.disconnect();
        });
        obs.observe(document.body, { childList: true, subtree: true });
        })();
        </script>
        """
        st.components.v1.html(js, height=0)

        # Đọc query param "place" theo API mới, fallback API cũ
        def _get_place():
            try:
                # Streamlit mới: trả str hoặc None
                val = st.query_params.get("place")
                return val
            except Exception:
                qp = st.experimental_get_query_params()
                v = qp.get("place")
                return v[0] if isinstance(v, list) and v else None

        def _clear_qp():
            try:
                st.query_params.clear()
            except Exception:
                st.experimental_set_query_params()

        raw = _get_place()
        if raw:
            try:
                x, y = map(int, raw.split(','))
                updated = False
                for i, m in enumerate(st.session_state.markers):
                    if m[0] == mid and m[1] < 0:
                        st.session_state.markers[i] = (mid, x, y, label)
                        updated = True
                        break
                if not updated:
                    st.session_state.markers.append((mid, x, y, label))
            finally:
                if 'pending_marker' in st.session_state:
                    del st.session_state.pending_marker
                _clear_qp()
                st.rerun()


with col2:
    st.write("**Hướng dẫn:**")
    st.write("1. Tải ảnh nền (nếu có)")
    st.write("2. Nhập ID & nhãn → 'Chuẩn bị đặt Marker'")
    st.write("3. **Click trên ảnh** để đặt")
    st.write("4. Thêm vùng câu trả lời trong tab **Vùng**")
    st.write("5. Lưu mẫu / Xuất file trong tab **Mẫu** & **Xuất**")
