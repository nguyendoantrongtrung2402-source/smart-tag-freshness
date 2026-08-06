import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang
st.set_page_config(
    page_title="Hệ Thống Phân Tích Độ Tươi Thực Phẩm",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-fresh {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-warning {
        background-color: #FEF08A;
        color: #713F12;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-spoiled {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Dữ liệu tham chiếu chuẩn pH - RGB
REF_DATA = [
    {"pH": 3.0, "rgb": (220, 60, 100), "freshness": 100},
    {"pH": 5.0, "rgb": (180, 80, 120), "freshness": 80},
    {"pH": 7.0, "rgb": (120, 100, 140), "freshness": 50},
    {"pH": 9.0, "rgb": (60, 120, 110), "freshness": 20},
    {"pH": 11.0, "rgb": (40, 140, 80), "freshness": 0}
]

def extract_median_rgb(image_np):
    """Cắt vùng trung tâm (ROI 40%) và tính median RGB"""
    h, w, _ = image_np.shape
    start_h, end_h = int(h * 0.3), int(h * 0.7)
    start_w, end_w = int(w * 0.3), int(w * 0.7)
    roi = image_np[start_h:end_h, start_w:end_w]
    
    med_r = int(np.median(roi[:, :, 0]))
    med_g = int(np.median(roi[:, :, 1]))
    med_b = int(np.median(roi[:, :, 2]))
    return (med_r, med_g, med_b), roi

def calculate_idw(target_rgb):
    """Thuật toán Inverse Distance Weighting (IDW) tính pH và Độ tươi"""
    weights = []
    eps = 1e-6
    for item in REF_DATA:
        dist = np.linalg.norm(np.array(target_rgb) - np.array(item["rgb"]))
        w = 1.0 / ((dist + eps) ** 2)
        weights.append(w)
    
    weights = np.array(weights)
    sum_w = np.sum(weights)
    
    est_ph = sum([item["pH"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    est_freshness = sum([item["freshness"] * w for item, w in zip(REF_DATA, weights)]) / sum_w
    return round(float(est_ph), 2), round(float(est_freshness), 1)

# 3. Thanh bên (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/test-tube.png", width=60)
    st.title("Thông Tin Dự Án")
    st.info("**Màng chỉ thị sinh học tía tô (Perilla Extract Film)**\n\nPhân tích độ tươi thực phẩm dựa trên sự thay đổi màu sắc chỉ thị pH.")
    
    st.divider()
    st.subheader("💡 Hướng dẫn")
    st.markdown("""
    1. Chụp/Tải ảnh màng chỉ thị.
    2. Đảm bảo màng nằm ở trung tâm ảnh.
    3. Xem kết quả phân tích tự động.
    """)
    st.divider()
    st.caption("Dự án Khoa Học Kỹ Thuật © 2026")

# 4. Giao diện chính
st.markdown('<div class="main-header">🍃 MÀNG CHỈ THỊ THÔNG MINH - KIỂM TRA ĐỘ TƯƠI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Định lượng độ tươi thực phẩm qua phân tích màu sắc màng tía tô bằng AI & Máy học</div>', unsafe_allow_html=True)

# Khung tải ảnh duy nhất (Tối ưu cho cả PC và ĐT)
uploaded_file = st.file_uploader(
    "📷 Chụp ảnh mới hoặc Chọn ảnh màng chỉ thị từ máy", 
    type=["jpg", "jpeg", "png"],
    help="Trên điện thoại, bấm vào đây để mở ứng dụng Camera chụp trực tiếp!"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    # Xử lý thuật toán
    rgb, roi = extract_median_rgb(img_np)
    ph, freshness = calculate_idw(rgb)
    
    st.divider()
    
    # Hiển thị kết quả dạng Cột
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Mẫu Phân Tích")
        st.image(image, caption="Ảnh gốc màng chỉ thị", use_container_width=True)
        
        # Bảng màu đại diện được trích xuất
        color_box = np.zeros((60, 300, 3), dtype=np.uint8)
        color_box[:] = rgb
        st.caption("Màu trung bình được trích xuất từ vùng trung tâm (ROI):")
        st.image(color_box, caption=f"RGB: {rgb}", use_container_width=True)

    with col2:
        st.subheader("📊 Kết Quả Đánh Giá")
        
        # Đánh giá trạng thái
        if freshness >= 75:
            status_html = '<div class="status-fresh">🟢 TƯƠI NGUYÊN - AN TOÀN SỬ DỤNG</div>'
        elif freshness >= 45:
            status_html = '<div class="status-warning">🟡 BẮT ĐẦU GIẢM TƯƠI - NÊN DÙNG NGAY</div>'
        else:
            status_html = '<div class="status-spoiled">🔴 CÓ DẤU HIỆU HỎNG - KHÔNG NÊN DÙNG</div>'
            
        st.markdown(status_html, unsafe_allow_html=True)
        st.write("")
        
        # Chỉ số 1: % Độ tươi
        st.metric(label="Chỉ số độ tươi (Freshness Index)", value=f"{freshness}%")
        st.progress(int(freshness) / 100)
        
        # Chỉ số 2: pH ước tính
        st.metric(label="Giá trị pH dự đoán", value=f"pH {ph}")
        
        # Chi tiết kỹ thuật
        with st.expander("🔍 Xem thông số kỹ thuật chi tiết"):
            st.write(f"- **Mã màu RGB:** `{rgb}`")
            st.write(f"- **Thuật toán nội suy:** IDW (Inverse Distance Weighting)")
            st.write(f"- **Vùng trích xuất (ROI):** Center 40%")

else:
    st.info("👆 Vui lòng bấm vào khung phía trên để chụp ảnh hoặc tải ảnh màng chỉ thị lên.")
