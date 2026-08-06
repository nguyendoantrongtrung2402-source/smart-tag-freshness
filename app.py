import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang web & Giao diện Mobile
st.set_page_config(
    page_title="Màng Chỉ Thị Tía Tô - Kiểm Tra Độ Tươi",
    page_icon="🍃",
    layout="centered"
)

# Thêm CSS tùy chỉnh cho các thẻ hiển thị trên điện thoại
st.markdown("""
    <style>
    .title-text {
        font-size: 1.6rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sub-text {
        font-size: 0.9rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-fresh { 
        background-color: #DEF7EC; 
        color: #03543F; 
        font-weight: bold; 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center;
        font-size: 1.1rem;
    }
    .status-warning { 
        background-color: #FEF08A; 
        color: #713F12; 
        font-weight: bold; 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center;
        font-size: 1.1rem;
    }
    .status-spoiled { 
        background-color: #FDE8E8; 
        color: #9B1C1C; 
        font-weight: bold; 
        padding: 12px; 
        border-radius: 8px; 
        text-align: center;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Dữ liệu tham chiếu màu thực nghiệm (Hệ PC:MF trích xuất từ hình ảnh thí nghiệm)
# Dữ liệu dạng: [Mốc pH, Mã RGB thực nghiệm (R, G, B), % Độ tươi tương ứng]
REF_DATA = [
    {"pH": 4.0,  "rgb": (185, 80, 120),  "freshness": 100}, # Axit nhạt / Màng tươi nguyên
    {"pH": 6.0,  "rgb": (145, 95, 135),  "freshness": 85},  # Tím nhạt / Thực phẩm an toàn
    {"pH": 7.0,  "rgb": (75, 70, 115),   "freshness": 50},  # Tím chàm thẫm / Nên sử dụng ngay
    {"pH": 8.0,  "rgb": (65, 115, 145),  "freshness": 15},  # Xanh lam / Bắt đầu hỏng (kiềm nhẹ)
    {"pH": 9.0,  "rgb": (50, 100, 110),  "freshness": 0}    # Xanh lục / Đã hỏng hoàn toàn
]

def extract_median_rgb(image_np):
    """Trích xuất mã màu RGB trung vị ở vùng trung tâm (ROI 40%) để khử nhiễu"""
    h, w, _ = image_np.shape
    start_h, end_h = int(h * 0.3), int(h * 0.7)
    start_w, end_w = int(w * 0.3), int(w * 0.7)
    roi = image_np[start_h:end_h, start_w:end_w]
    
    med_r = int(np.median(roi[:, :, 0]))
    med_g = int(np.median(roi[:, :, 1]))
    med_b = int(np.median(roi[:, :, 2]))
    return (med_r, med_g, med_b), roi

def calculate_idw(target_rgb):
    """Thuật toán IDW (Inverse Distance Weighting) tính pH và % Độ tươi"""
    weights = []
    eps = 1e-6
    for item in REF_DATA:
        # Tính khoảng cách Euclidean giữa màu chụp được và màu mẫu
        dist = np.linalg.norm(np.array(target_rgb) - np.array(item["rgb"]))
        w = 1.0 / ((dist + eps) ** 2)
        weights.append(w)
    
    weights = np.array(weights)
    sum_w = np.sum(weights)
    
    # Nội suy giá trị pH và độ tươi
    est_ph = sum([item["pH"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    est_freshness = sum([item["freshness"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    return round(float(est_ph), 2), round(float(est_freshness), 1)

# 3. Giao diện người dùng (UI)
st.markdown('<div class="title-text">🍃 MÀNG CHỈ THỊ TÍA TÔ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Ứng dụng định lượng độ tươi thực phẩm qua phân tích màu sắc</div>', unsafe_allow_html=True)

# Khung tải ảnh / chụp ảnh
uploaded_file = st.file_uploader(
    "📷 Chụp ảnh hoặc tải ảnh màng chỉ thị lên:", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Đọc ảnh và chuyển sang định dạng RGB
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    # Tính toán kết quả thông qua thuật toán IDW
    rgb, roi = extract_median_rgb(img_np)
    ph, freshness = calculate_idw(rgb)
    
    # --- HIỂN THỊ KẾT QUẢ NGAY PHÍA TRÊN ---
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ PHÂN TÍCH")
    
    # Cảnh báo trạng thái theo độ tươi
    if freshness >= 70:
        st.markdown('<div class="status-fresh">🟢 THỰC PHẨM TƯƠI - AN TOÀN</div>', unsafe_allow_html=True)
    elif freshness >= 40:
        st.markdown('<div class="status-warning">🟡 BẮT ĐẦU GIẢM TƯƠI - NÊN DÙNG NGAY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-spoiled">🔴 THỰC PHẨM ĐÃ HỎNG - KHÔNG DÙNG</div>', unsafe_allow_html=True)
    
    # Thẻ chỉ số % Độ tươi và pH
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Độ tươi", f"{freshness}%")
    with c2:
        st.metric("Dự đoán pH", f"pH {ph}")
        
    st.progress(int(freshness) / 100)
    
    # --- PHẦN CHI TIẾT KỸ THUẬT (ẨN TRONG EXPANDER) ---
    with st.expander("🔍 Xem chi tiết ảnh & mã màu trích xuất"):
        st.image(image, caption="Ảnh màng chỉ thị đã chụp", use_container_width=True)
        
        # Tạo khối hiển thị màu trung vị trích xuất được
        color_box = np.zeros((40, 300, 3), dtype=np.uint8)
        color_box[:] = rgb
        st.caption(f"Màu trung vị vùng trung tâm (RGB: {rgb}):")
        st.image(color_box, use_container_width=True)

# Thông tin phụ ở thanh Sidebar
with st.sidebar:
    st.header("⚙️ Thông tin dự án")
    st.write("Ứng dụng định lượng độ tươi thực phẩm dựa trên thuật toán IDW và bảng màu chuyển hóa Anthocyanin từ lá tía tô.")
    st.caption("Dự án Khoa học Kỹ thuật © 2026")
