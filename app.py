import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang Dark Mode
st.set_page_config(
    page_title="SaveMyFood - Tía Tô Indicator",
    page_icon="🥑",
    layout="centered"
)

# 2. Custom CSS Phong cách SaveMyFood (Dark Mode / Neon Green)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0B140E !important;
        color: #ECFDF5;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header */
    .brand-title {
        color: #10B981;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .greeting {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    /* Main Hero Card (Chỉ số chính) */
    .hero-card {
        background: linear-gradient(145deg, #132219 0%, #1A2E22 100%);
        border: 1px solid #1F3A2B;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    }
    
    .card-label {
        color: #8E9BAE;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .score-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
    }
    .score-unit {
        font-size: 1.8rem;
        color: #10B981;
    }
    
    /* Badge trạng thái */
    .badge-excellent {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 12px;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 12px;
    }
    .badge-spoiled {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 12px;
    }

    /* Sub Cards (Lưới 2 cột) */
    .sub-card {
        background: #132219;
        border: 1px solid #1F3A2B;
        border-radius: 20px;
        padding: 18px;
        text-align: center;
    }
    .sub-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #34D399;
    }
    
    /* Custom Upload Box */
    .stFileUploader {
        background: #132219;
        border: 2px dashed #10B981;
        border-radius: 20px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. REF_DATA theo dải pH = 3, 5, 7, 9, 11 (Cập nhật màu thực nghiệm Tía tô)
REF_DATA = [
    {"pH": 3.0,  "rgb": (205, 50, 90),   "freshness": 100}, # Axit mạnh / Đỏ hồng tươi
    {"pH": 5.0,  "rgb": (165, 75, 125),  "freshness": 90},  # Axit nhẹ / Hồng tím (Rất tươi)
    {"pH": 7.0,  "rgb": (75, 65, 115),   "freshness": 50},  # Trung tính / Tím chàm (Cần dùng)
    {"pH": 9.0,  "rgb": (45, 110, 110),  "freshness": 10},  # Kiềm nhẹ / Xanh lục (Ươn/Hỏng)
    {"pH": 11.0, "rgb": (80, 130, 70),   "freshness": 0}    # Kiềm mạnh / Xanh vàng (Hỏng nặng)
]

def extract_median_rgb(image_np):
    h, w, _ = image_np.shape
    roi = image_np[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    return (int(np.median(roi[:, :, 0])), int(np.median(roi[:, :, 1])), int(np.median(roi[:, :, 2]))), roi

def calculate_idw(target_rgb):
    eps = 1e-6
    weights = [1.0 / ((np.linalg.norm(np.array(target_rgb) - np.array(item["rgb"])) + eps) ** 2) for item in REF_DATA]
    sum_w = sum(weights)
    est_ph = sum([item["pH"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    est_freshness = sum([item["freshness"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    return round(float(est_ph), 2), round(float(est_freshness), 1)

# 4. Header UI
st.markdown('<div class="brand-title">SaveMyFood 🥑</div>', unsafe_allow_html=True)
st.markdown('<div class="greeting">Kiểm tra độ tươi thực phẩm</div>', unsafe_allow_html=True)

# 5. Khung tải / Chụp ảnh
uploaded_file = st.file_uploader(
    "Quét màng chỉ thị tía tô:", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    rgb, roi = extract_median_rgb(img_np)
    ph, freshness = calculate_idw(rgb)
    
    # Xác định Badge & Text trạng thái
    if freshness >= 70:
        badge_html = '<div class="badge-excellent">🟢 Excellent (Tươi ngon)</div>'
    elif freshness >= 40:
        badge_html = '<div class="badge-warning">🟡 Warning (Nên dùng ngay)</div>'
    else:
        badge_html = '<div class="badge-spoiled">🔴 Spoiled (Thực phẩm hỏng)</div>'
        
    st.write("")
    
    # HERO CARD KẾT QUẢ (GIỐNG SAVEMYFOOD)
    st.markdown(f"""
        <div class="hero-card">
            <div class="card-label">Freshness Score</div>
            <div class="score-container">
                <div>
                    <span class="score-number">{int(freshness)}</span>
                    <span class="score-unit">%</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.5rem;">🍃</div>
                </div>
            </div>
            {badge_html}
        </div>
    """, unsafe_allow_html=True)
    
    # LƯỚI THÔNG SỐ PHỤ (2 CỘT)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="sub-card">
                <div class="card-label">DỰ ĐOÁN pH</div>
                <div class="sub-val">{ph}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="sub-card">
                <div class="card-label">MÃ MÀU RGB</div>
                <div class="sub-val" style="font-size: 1rem; color: #94A3B8; padding-top: 10px;">
                    {rgb}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # CHI TIẾT ẢNH
    with st.expander("🔍 Chi tiết ảnh chụp & Bảng màu"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(image, caption="Ảnh màng gốc", use_container_width=True)
        with col_b:
            color_box = np.zeros((100, 200, 3), dtype=np.uint8)
            color_box[:] = rgb
            st.image(color_box, caption="Màu trích xuất ROI", use_container_width=True)

else:
    # Trạng thái chờ tải ảnh
    st.info("👆 Hãy chụp hoặc chọn ảnh màng chỉ thị tía tô để tiến hành phân tích.")
