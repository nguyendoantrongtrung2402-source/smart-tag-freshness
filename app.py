import streamlit as st
import numpy as np
from PIL import Image

# Hàm làm sạch chuỗi HTML/CSS loại bỏ lùi đầu dòng thừa
def _html(s: str) -> str:
    return "\n".join(line.strip() for line in s.strip().split("\n"))

# -----------------------------------------------------------------------------
# 1. BẢNG DỮ LIỆU HIỆU CHUẨN MÀU HÓA HỌC (RGB vs pH)
# -----------------------------------------------------------------------------
REF_DATA = [
    {"pH": 3.0,  "rgb": (220, 60, 80)},
    {"pH": 5.0,  "rgb": (210, 110, 140)},
    {"pH": 7.0,  "rgb": (140, 120, 170)},
    {"pH": 9.0,  "rgb": (90, 140, 160)},
    {"pH": 11.0, "rgb": (50, 120, 100)}
]
CALIBRATION_TARGET_RGB = np.array([200.0, 200.0, 200.0])

# -----------------------------------------------------------------------------
# 2. HÀM TÍNH TOÁN BACKEND
# -----------------------------------------------------------------------------
def pH_to_freshness(ph, ph_fresh=6.0, ph_spoiled=8.5):
    if ph <= ph_fresh:
        return 100.0
    elif ph >= ph_spoiled:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (ph_spoiled - ph) / (ph_spoiled - ph_fresh)))

def calculate_idw(target_rgb):
    eps = 1e-6
    target_arr = np.array(target_rgb)
    weights = [1.0 / ((np.linalg.norm(target_arr - np.array(item["rgb"])) + eps) ** 2) for item in REF_DATA]
    sum_w = sum(weights)
    est_ph = sum(item["pH"] * w for item, w in zip(REF_DATA, weights)) / sum_w
    return round(float(est_ph), 2), round(float(pH_to_freshness(est_ph)), 1)

def extract_center_rgb_median(image: Image.Image, crop_ratio=0.3):
    img_arr = np.array(image.convert("RGB"))
    h, w, _ = img_arr.shape
    ch_s, ch_e = int(h * (0.5 - crop_ratio/2)), int(h * (0.5 + crop_ratio/2))
    cw_s, cw_e = int(w * (0.5 - crop_ratio/2)), int(w * (0.5 + crop_ratio/2))
    region = img_arr[ch_s:ch_e, cw_s:cw_e]
    return tuple(map(int, np.median(region, axis=(0, 1))))

def extract_card_rgb_median(image: Image.Image, size_ratio=0.15):
    """Cắt vùng góc trên-trái (15% chiều rộng/cao) chứa thẻ tham chiếu màu"""
    img_arr = np.array(image.convert("RGB"))
    h, w, _ = img_arr.shape
    region = img_arr[0:int(h*size_ratio), 0:int(w*size_ratio)]
    return tuple(map(int, np.median(region, axis=(0, 1))))

def apply_color_calibration(measured_rgb, card_rgb):
    eps = 1e-6
    scale = CALIBRATION_TARGET_RGB / (np.array(card_rgb, dtype=float) + eps)
    calibrated = np.clip(np.array(measured_rgb, dtype=float) * scale, 0, 255)
    return tuple(map(int, calibrated))

# -----------------------------------------------------------------------------
# 3. VÒNG TRÒN TIẾN ĐỘ (SVG)
# -----------------------------------------------------------------------------
def render_progress_ring(percent, color_hex, size=132, stroke=10):
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - percent / 100)
    return _html(f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform: rotate(-90deg); width: 100%; height: 100%;">
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke}" />
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{color_hex}" stroke-width="{stroke}" stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" style="transition: stroke-dashoffset 0.6s ease;" />
    </svg>
    """)

# -----------------------------------------------------------------------------
# 4. CẤU HÌNH TRANG & THIẾT KẾ MỚI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SaveMyFood - Quản Lý Độ Tươi Thực Phẩm",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #0F2318 0%, #0A130E 45%, #08100C 100%);
        color: #EAF6EE;
    }
    #MainMenu, header, footer { visibility: hidden; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 480px; }

    .app-title {
        font-size: 1.7rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #3DDC84, #A7F3C4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .app-subtitle { color: #7C9285; font-size: 0.92rem; margin-bottom: 22px; }

    .glass-card {
        background: linear-gradient(155deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 16px;
        backdrop-filter: blur(6px);
    }

    .hero-flex { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .hero-label { color: #7C9285; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
    .hero-score { font-size: 3.2rem; font-weight: 800; line-height: 1; }
    .hero-score-unit { font-size: 1.4rem; font-weight: 600; opacity: 0.6; margin-left: 2px; }

    .ring-wrap { position: relative; width: 132px; height: 132px; display: flex; align-items: center; justify-content: center; }
    .ring-icon { position: absolute; font-size: 1.8rem; }

    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 7px 16px; border-radius: 999px;
        font-size: 0.85rem; font-weight: 700; margin-top: 14px;
    }
    .pill-fresh   { background: rgba(61, 220, 132, 0.14); color: #3DDC84; }
    .pill-warning { background: rgba(255, 176, 32, 0.14); color: #FFB020; }
    .pill-danger  { background: rgba(255, 82, 82, 0.14); color: #FF5252; }

    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .metric-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 14px 16px; }
    .metric-box .label { color: #7C9285; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 6px; }
    .metric-box .value { font-size: 1.35rem; font-weight: 700; }

    .swatch { width: 22px; height: 22px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.25); display: inline-block; vertical-align: middle; margin-right: 8px; }

    div[data-testid="stCameraInput"] > label, div[data-testid="stFileUploader"] > label { color: #7C9285 !important; }
    div[data-testid="stCameraInput"], div[data-testid="stFileUploader"] {
        border: 1.5px dashed rgba(61, 220, 132, 0.35);
        border-radius: 18px; padding: 6px;
        background: rgba(61, 220, 132, 0.03);
    }

    @media (max-width: 420px) {
        .hero-score { font-size: 2.5rem; }
        .ring-wrap { width: 104px; height: 104px; }
        .glass-card { padding: 16px; }
    }
</style>
"""), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title">🌿 SaveMyFood</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Định lượng độ tươi thực phẩm qua màng chỉ thị tía tô &amp; phân tích RGB</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. KHUNG NHẬP ẢNH
# -----------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="hero-label">📷 Thu thập hình ảnh</div>', unsafe_allow_html=True)

input_mode = st.radio("Chọn phương thức nhập ảnh:", ["Chụp ảnh trực tiếp", "Tải ảnh từ thư viện"],
                       horizontal=True, label_visibility="collapsed")

uploaded_image = None
if input_mode == "Chụp ảnh trực tiếp":
    camera_file = st.camera_input("Chụp màng chỉ thị trên nắp hộp", label_visibility="collapsed")
    if camera_file is not None:
        uploaded_image = Image.open(camera_file)
else:
    file_uploaded = st.file_uploader("Chọn ảnh màng chỉ thị...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file_uploaded is not None:
        uploaded_image = Image.open(file_uploaded)

use_calibration = st.checkbox("⚙️ Hiệu chỉnh màu tự động bằng thẻ tham chiếu")
if use_calibration:
    st.caption("📌 Lưu ý: Hãy đảm bảo thẻ màu chuẩn được đặt ở góc trên-trái khung hình khi chụp.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. KẾT QUẢ
# -----------------------------------------------------------------------------
if uploaded_image is not None:
    raw_r, raw_g, raw_b = extract_center_rgb_median(uploaded_image)

    if use_calibration:
        card_rgb = extract_card_rgb_median(uploaded_image)
        r, g, b = apply_color_calibration((raw_r, raw_g, raw_b), card_rgb)
    else:
        r, g, b = raw_r, raw_g, raw_b

    est_ph, est_freshness = calculate_idw((r, g, b))

    if est_freshness >= 75.0:
        pill_class, ring_color, status_text, icon = "pill-fresh", "#3DDC84", "Tươi mới, an toàn", "🍃"
    elif est_freshness >= 25.0:
        pill_class, ring_color, status_text, icon = "pill-warning", "#FFB020", "Cần dùng ngay", "⚠️"
    else:
        pill_class, ring_color, status_text, icon = "pill-danger", "#FF5252", "Đã hỏng, không nên dùng", "⛔"

    ring_svg = render_progress_ring(est_freshness, ring_color)

    st.markdown(_html(f"""
    <div class="glass-card">
    <div class="hero-flex">
    <div>
    <div class="hero-label">Chỉ số độ tươi</div>
    <div><span class="hero-score">{est_freshness:g}</span><span class="hero-score-unit">%</span></div>
    <div class="status-pill {pill_class}">{icon} {status_text}</div>
    </div>
    <div class="ring-wrap">
    {ring_svg}
    <div class="ring-icon">{icon}</div>
    </div>
    </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(_html(f"""
    <div class="glass-card">
    <div class="hero-label" style="margin-bottom: 12px;">🔬 Chi tiết phân tích</div>
    <div class="metric-grid">
    <div class="metric-box">
    <div class="label">pH dự đoán (IDW)</div>
    <div class="value">{est_ph}</div>
    </div>
    <div class="metric-box">
    <div class="label">Mã màu RGB</div>
    <div class="value" style="font-size: 1rem;">
    <span class="swatch" style="background: rgb({r},{g},{b});"></span>{r}, {g}, {b}
    </div>
    </div>
    </div>
    </div>
    """), unsafe_allow_html=True)

    with st.expander("Xem ảnh gốc đã chụp"):
        st.image(uploaded_image, use_container_width=True)
