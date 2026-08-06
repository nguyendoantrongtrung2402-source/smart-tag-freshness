import streamlit as st
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. BẢNG DỮ LIỆU HIỆU CHUẨN MÀU HÓA HỌC (RGB vs pH)
# Dùng thuần túy để hiệu chuẩn màu sắc với chỉ số pH (không chứa % độ tươi)
# -----------------------------------------------------------------------------
REF_DATA = [
    {"pH": 3.0,  "rgb": (220, 60, 80)},    # Đỏ acid
    {"pH": 5.0,  "rgb": (210, 110, 140)},  # Hồng nhạt
    {"pH": 7.0,  "rgb": (140, 120, 170)},  # Tím trung tính
    {"pH": 9.0,  "rgb": (90, 140, 160)},   # Xanh lam kiềm nhẹ
    {"pH": 11.0, "rgb": (50, 120, 100)}    # Xanh lục kiềm đậm
]

# Giá trị RGB tham chiếu chuẩn của ô thẻ màu trắng/xám (Ví dụ: Xám chuẩn 200,200,200)
CALIBRATION_TARGET_RGB = np.array([200.0, 200.0, 200.0])

# -----------------------------------------------------------------------------
# 2. HÀM TÍNH TOÁN BACKEND
# -----------------------------------------------------------------------------
def pH_to_freshness(ph, ph_fresh=6.0, ph_spoiled=8.5):
    """
    Quy đổi pH sang % độ tươi dựa trên dải sinh học thực tế của ức gà.
    Lưu ý: ph_fresh và ph_spoiled có thể điều chỉnh sau khi có số liệu thực nghiệm.
    """
    if ph <= ph_fresh:
        return 100.0
    elif ph >= ph_spoiled:
        return 0.0
    else:
        freshness = 100.0 * (ph_spoiled - ph) / (ph_spoiled - ph_fresh)
        return max(0.0, min(100.0, freshness))

def calculate_idw(target_rgb):
    """
    Sử dụng thuật toán IDW (Inverse Distance Weighting) tính pH từ màu RGB,
    sau đó quy đổi ra % độ tươi.
    """
    eps = 1e-6
    target_arr = np.array(target_rgb)
    
    weights = []
    for item in REF_DATA:
        ref_rgb = np.array(item["rgb"])
        dist = np.linalg.norm(target_arr - ref_rgb)
        w = 1.0 / ((dist + eps) ** 2)
        weights.append(w)
        
    sum_w = sum(weights)
    
    # Nội suy pH
    est_ph = sum([item["pH"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    
    # Tính độ tươi theo dải sinh học ức gà
    est_freshness = pH_to_freshness(est_ph)
    
    return round(float(est_ph), 2), round(float(est_freshness), 1)

def extract_center_rgb_median(image: Image.Image, crop_ratio=0.3):
    """
    Trích xuất giá trị RGB TRUNG VỊ (MEDIAN) từ vùng tâm ảnh màng chỉ thị
    để chống nhiễu do lóa sáng hoặc bóng đổ (Khớp với mục 3.5 trong Báo cáo).
    """
    img_arr = np.array(image.convert("RGB"))
    h, w, _ = img_arr.shape
    
    ch_start, ch_end = int(h * (0.5 - crop_ratio/2)), int(h * (0.5 + crop_ratio/2))
    cw_start, cw_end = int(w * (0.5 - crop_ratio/2)), int(w * (0.5 + crop_ratio/2))
    
    center_region = img_arr[ch_start:ch_end, cw_start:cw_end]
    
    # Dùng Trung vị (Median) thay vì Trung bình cộng (Mean)
    median_rgb = np.median(center_region, axis=(0, 1))
    return tuple(map(int, median_rgb))

def apply_color_calibration(measured_rgb, card_measured_rgb):
    """
    Hiệu chỉnh màu sắc dựa trên ô màu chuẩn trên Thẻ tham chiếu (Color Calibration Card).
    Giúp chống nhiễu do thuật toán Auto-White-Balance (AWB) của camera điện thoại.
    """
    eps = 1e-6
    card_arr = np.array(card_measured_rgb, dtype=float) + eps
    
    # Tính hệ số nhân cân bằng màu cho từng kênh R, G, B
    scale_factors = CALIBRATION_TARGET_RGB / card_arr
    
    # Áp dụng hệ số vào màu màng đo được
    calibrated_rgb = np.array(measured_rgb, dtype=float) * scale_factors
    calibrated_rgb = np.clip(calibrated_rgb, 0, 255) # Giới hạn dải 0-255
    
    return tuple(map(int, calibrated_rgb))

# -----------------------------------------------------------------------------
# 3. CẤU HÌNH TRANG & STYLESHEET (DARK MODE / SAVE MY FOOD STYLE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SaveMyFood - Quản Lý Độ Tươi Thực Phẩm",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Nền tối chủ đạo */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Card giao diện chính */
    .custom-card {
        background-color: #1A1C23;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid #2D313E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Hiển thị điểm số nổi bật */
    .score-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
    }
    
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .score-unit {
        font-size: 1.5rem;
        font-weight: 600;
        color: #888888;
    }
    
    /* Badge trạng thái */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        text-align: center;
    }
    .badge-fresh { background-color: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; }
    .badge-warning { background-color: rgba(255, 171, 0, 0.15); color: #FFAB00; border: 1px solid #FFAB00; }
    .badge-danger { background-color: rgba(255, 23, 68, 0.15); color: #FF1744; border: 1px solid #FF1744; }

    /* CSS Responsive cho màn hình di động hẹp (< 480px) */
    @media (max-width: 480px) {
        .score-number {
            font-size: 2.4rem !important;
        }
        .score-unit {
            font-size: 1.1rem !important;
        }
        .custom-card {
            padding: 14px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. GIAO DIỆN NGUỜI DÙNG (MOBILE-FIRST UI)
# -----------------------------------------------------------------------------
st.title("🍃 SaveMyFood")
st.caption("Định lượng độ tươi thực phẩm qua màng chỉ thị Tía tô & AI RGB")

st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.subheader("📷 Thu thập hình ảnh màng chỉ thị")

input_mode = st.radio(
    "Chọn phương thức nhập ảnh:",
    ["Chụp ảnh trực tiếp", "Tải ảnh từ thư viện"],
    horizontal=True
)

uploaded_image = None

if input_mode == "Chụp ảnh trực tiếp":
    camera_file = st.camera_input("Chụp màng chỉ thị trên nắp hộp")
    if camera_file is not None:
        uploaded_image = Image.open(camera_file)
else:
    file_uploaded = st.file_uploader("Chọn ảnh màng chỉ thị...", type=["jpg", "jpeg", "png"])
    if file_uploaded is not None:
        uploaded_image = Image.open(file_uploaded)

# Tùy chọn bật/tắt Hiệu chỉnh thẻ màu chuẩn
use_calibration = st.checkbox("⚙️ Kích hoạt Hiệu chỉnh màu với Thẻ màu tham chiếu (Calibration Card)")
card_rgb_input = (200, 200, 200)

if use_calibration:
    st.info("Nhập giá trị RGB đo được từ ô màu xám/trắng trên thẻ tham chiếu trong ảnh:")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        c_r = st.number_input("R (Thẻ)", 0, 255, 190)
    with col_c2:
        c_g = st.number_input("G (Thẻ)", 0, 255, 190)
    with col_c3:
        c_b = st.number_input("B (Thẻ)", 0, 255, 190)
    card_rgb_input = (c_r, c_g, c_b)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
# -----------------------------------------------------------------------------
if uploaded_image is not None:
    # 1. Trích xuất màu Trung vị (Median RGB)
    raw_r, raw_g, raw_b = extract_center_rgb_median(uploaded_image)
    
    # 2. Áp dụng hiệu chỉnh thẻ màu (nếu bật)
    if use_calibration:
        r, g, b = apply_color_calibration((raw_r, raw_g, raw_b), card_rgb_input)
    else:
        r, g, b = raw_r, raw_g, raw_b

    # 3. Tính pH (IDW) & % Độ tươi
    est_ph, est_freshness = calculate_idw((r, g, b))
    
    # 4. Xác định trạng thái & Badge đồng bộ chuẩn với Thang điểm Cảm quan 5 Mức (Bước 25%)
    if est_freshness >= 75.0:       # Tương ứng Mức 4 - Mức 5 (75% - 100%)
        status_text = "TƯƠI MỚI - AN TOÀN"
        badge_class = "badge-fresh"
        theme_color = "#00E676"
    elif est_freshness >= 25.0:     # Tương ứng Mức 2 - Mức 3 (25% - 74.9%)
        status_text = "CẦN CHÚ Ý - DÙNG NGAY"
        badge_class = "badge-warning"
        theme_color = "#FFAB00"
    else:                           # Tương ứng Mức 1 (0% - 24.9%)
        status_text = "ĐÃ HỎNG - KHÔNG NÊN DÙNG"
        badge_class = "badge-danger"
        theme_color = "#FF1744"

    # --- KHỐI HIỂN THỊ KẾT QUẢ ---
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='status-badge {badge_class}'>{status_text}</span>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='score-container'>
        <div>
            <div style='color: #888888; font-size: 0.9rem; margin-top: 8px;'>Chỉ số độ tươi</div>
            <span class='score-number' style='color: {theme_color};'>{est_freshness}</span>
            <span class='score-unit'>%</span>
        </div>
        <div style='text-align: right;'>
            <div style='color: #888888; font-size: 0.85rem;'>Màu đã trích xuất (Median)</div>
            <div style='width: 36px; height: 36px; background-color: rgb({r},{g},{b}); border-radius: 50%; border: 2px solid #FFF; display: inline-block; margin-top: 4px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(int(est_freshness))
    st.markdown("</div>", unsafe_allow_html=True)

    # --- KHỐI CHI TIẾT THÔNG SỐ KĨ THUẬT ---
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Chi tiết phân tích")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="pH Dự đoán (IDW)", value=f"{est_ph}")
    with col2:
        st.metric(label="Mã màu RGB đã xử lý", value=f"{r}, {g}, {b}")
        
    if use_calibration:
        st.caption(f"📍 Màu thô gốc (Chưa cân bằng): RGB({raw_r}, {raw_g}, {raw_b})")
        
    st.caption("Ứng dụng thuật toán IDW nội suy pH từ màu sắc trung vị, quy đổi ra độ tươi theo dải sinh học ức gà (pH 6.0 - 8.5).")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.image(uploaded_image, caption="Ảnh màng chỉ thị đã xử lý", use_container_width=True)
