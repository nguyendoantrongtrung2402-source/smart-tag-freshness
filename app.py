import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="Hệ thống Đánh giá Độ tươi",
    page_icon="🥩",
    layout="centered"
)

# Bảng màu chuẩn từ thí nghiệm
REF_DATA = {
    3:  {"rgb": np.array([210, 80, 100]), "label": "Rất tươi"},
    5:  {"rgb": np.array([190, 95, 120]), "label": "Tươi"},
    7:  {"rgb": np.array([150, 110, 130]), "label": "Bắt đầu biến chất"},
    9:  {"rgb": np.array([100, 130, 120]), "label": "Ương/Cần dùng ngay"},
    11: {"rgb": np.array([60,  140, 90]),  "label": "Hỏng hoàn toàn"}
}

def extract_median_rgb(image_pil):
    """Trích xuất màu RGB trung vị từ ảnh"""
    img_np = np.array(image_pil)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    h, w, _ = img_np.shape
    roi = img_np[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    
    return np.array([int(np.median(roi[:, :, 0])), int(np.median(roi[:, :, 1])), int(np.median(roi[:, :, 2]))]), img_np

def predict_ph_and_freshness(sample_rgb, power=2):
    """Thuật toán IDW tính toán độ tươi"""
    weights = {}
    for ph, data in REF_DATA.items():
        dist = np.linalg.norm(sample_rgb - data["rgb"])
        if dist == 0:
            dist = 1e-6
        weights[ph] = 1.0 / (dist ** power)
        
    total_weight = sum(weights.values())
    estimated_ph = sum(ph * (weights[ph] / total_weight) for ph in REF_DATA.keys())
    
    if estimated_ph <= 7.0:
        freshness_pct = 100.0
    elif estimated_ph >= 11.0:
        freshness_pct = 0.0
    else:
        freshness_pct = 100.0 - ((estimated_ph - 7.0) / (11.0 - 7.0)) * 100.0
        
    return estimated_ph, freshness_pct

# --- GIAO DIỆN WEB APP ---
st.title("🥩 Smart Tag - Đánh Giá Độ Tươi")
st.caption("Dự án KHKT: Màng chỉ thị sinh học tía tô & Thuật toán IDW")
st.markdown("---")

source_option = st.radio(
    "Chọn phương thức tải ảnh:",
    ("📸 Chụp bằng Camera", "📁 Tải ảnh từ thiết bị"),
    horizontal=True
)

if "Chụp" in source_option:
    image_file = st.camera_input("Chụp trực tiếp màng chỉ thị")
else:
    image_file = st.file_uploader("Chọn ảnh màng màu (.jpg, .png)", type=["jpg", "jpeg", "png"])

if image_file is not None:
    pil_img = Image.open(image_file)
    sample_rgb, img_np = extract_median_rgb(pil_img)
    est_ph, freshness = predict_ph_and_freshness(sample_rgb)
    
    st.markdown("### 📊 Kết quả phân tích")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(pil_img, caption="Ảnh màng nhận diện", use_container_width=True)
        
    with col2:
        st.write("**Thông số màu trích xuất (RGB):**")
        st.code(f"R: {sample_rgb[0]} | G: {sample_rgb[1]} | B: {sample_rgb[2]}")
        hex_color = f"#{sample_rgb[0]:02x}{sample_rgb[1]:02x}{sample_rgb[2]:02x}"
        st.markdown(f"**Màu trung vị:** <div style='width:100%; height:30px; background-color:{hex_color}; border-radius:5px; border:1px solid #ccc;'></div>", unsafe_allow_html=True)
        st.write(f"🧪 **pH ước tính:** `{est_ph:.2f}`")
    
    st.markdown("#### Chỉ số độ tươi:")
    st.progress(int(freshness) / 100)
    st.write(f"### **{freshness:.1f}%**")
    
    if freshness >= 80:
        st.success("✅ **Thực phẩm rất tươi.** Đảm bảo chất lượng an toàn để chế biến.")
    elif freshness >= 50:
        st.warning("⚠️ **Thực phẩm bắt đầu giảm độ tươi.** Nên chế biến ngay.")
    else:
        st.error("🚨 **Thực phẩm đã hỏng / Ương.** Không nên tiếp tục sử dụng!")
