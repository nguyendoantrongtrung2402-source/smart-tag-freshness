import os
import colorsys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ============================================================
# FRESHTAG - ỨNG DỤNG PHÂN LOẠI TRẠNG THÁI ỨC GÀ
# Thẻ chỉ thị chitosan - tía tô + ảnh smartphone + mô hình ML
#
# LƯU Ý KHOA HỌC
# - Không hiển thị "% độ tươi".
# - Không khẳng định "an toàn để ăn".
# - Đầu ra chính: 3 trạng thái.
# - Nếu chưa có model.joblib, app chạy ở CHẾ ĐỘ DEMO để thử giao diện.
# ============================================================


# -----------------------------
# 1. CẤU HÌNH ỨNG DỤNG
# -----------------------------
APP_NAME = "FreshTag"
APP_SUBTITLE = "Phân loại trạng thái ức gà bằng thẻ chỉ thị tía tô"

MODEL_PATH = Path("model.joblib")

# ROI tính theo tỉ lệ ảnh: (x1, y1, x2, y2), từ 0 đến 1.
# Hãy chỉnh các giá trị này sau khi vị trí camera + thẻ trong hộp chụp đã cố định.
INDICATOR_ROI = (0.35, 0.35, 0.65, 0.65)

# Vùng thẻ xám/thẻ tham chiếu. Mặc định nằm ở góc trên-trái.
GRAY_CARD_ROI = (0.03, 0.03, 0.18, 0.18)

# Màu mục tiêu của thẻ xám dùng để cân chỉnh đơn giản.
# Có thể thay đổi sau khi nhóm chốt thẻ tham chiếu thật.
GRAY_TARGET_RGB = np.array([200.0, 200.0, 200.0], dtype=float)

# Thứ tự đặc trưng phải trùng với lúc huấn luyện model.
FEATURE_COLUMNS = ["R", "G", "B", "H", "S", "V"]

# Tên lớp chuẩn của đề tài.
CLASS_LABELS = {
    "fresh": "CÒN TƯƠI",
    "transition": "NÊN SỬ DỤNG SỚM",
    "spoiled": "CÓ DẤU HIỆU HƯ HỎNG",
}

# Demo centroids CHỈ để chạy thử UI trước khi có dữ liệu thật.
# PHẢI thay bằng model.joblib đã huấn luyện từ dữ liệu thí nghiệm.
DEMO_CENTROIDS = {
    "fresh": np.array([205.0, 95.0, 135.0]),
    "transition": np.array([145.0, 110.0, 165.0]),
    "spoiled": np.array([75.0, 135.0, 125.0]),
}


# -----------------------------
# 2. CẤU HÌNH STREAMLIT
# -----------------------------
st.set_page_config(
    page_title=f"{APP_NAME} - Thẻ chỉ thị tía tô",
    page_icon="🟣",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# 3. CSS
# -----------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #0d0a12;
            --panel: #17111f;
            --panel-2: #20162a;
            --text: #f7f2fb;
            --muted: #a99daf;
            --purple: #b978ff;
            --purple-2: #8e44d6;
            --fresh: #43c982;
            --warning: #f2b84b;
            --danger: #ef6262;
            --border: rgba(255,255,255,.10);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(116, 68, 148, .22), transparent 32%),
                linear-gradient(180deg, #100b16 0%, #0b0910 100%);
            color: var(--text);
        }

        #MainMenu, header, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 660px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, #b978ff, #6f35a9);
            font-size: 23px;
            box-shadow: 0 10px 30px rgba(132, 71, 178, .25);
        }

        .brand-name {
            font-size: 1.75rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            line-height: 1;
        }

        .subtitle {
            color: var(--muted);
            font-size: .95rem;
            margin: 8px 0 22px 54px;
        }

        .card {
            background: linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px;
            margin: 14px 0;
        }

        .section-kicker {
            color: #c9b9d2;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .helper {
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.55;
        }

        .result-card {
            border-radius: 24px;
            padding: 24px 22px;
            border: 1px solid var(--border);
            margin-top: 18px;
        }

        .result-fresh {
            background: linear-gradient(160deg, rgba(67,201,130,.16), rgba(67,201,130,.04));
        }

        .result-transition {
            background: linear-gradient(160deg, rgba(242,184,75,.16), rgba(242,184,75,.04));
        }

        .result-spoiled {
            background: linear-gradient(160deg, rgba(239,98,98,.16), rgba(239,98,98,.04));
        }

        .result-label {
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .result-main {
            margin-top: 6px;
            font-size: 1.62rem;
            font-weight: 900;
            letter-spacing: -.03em;
        }

        .result-desc {
            margin-top: 8px;
            color: #d4c9da;
            line-height: 1.55;
            font-size: .94rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }

        .metric {
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 16px;
            padding: 13px 14px;
        }

        .metric-label {
            font-size: .72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .06em;
            font-weight: 750;
        }

        .metric-value {
            font-size: 1rem;
            font-weight: 800;
            margin-top: 5px;
            overflow-wrap: anywhere;
        }

        .swatch {
            display: inline-block;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,.32);
            vertical-align: -4px;
            margin-right: 7px;
        }

        .flow {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            align-items: center;
            gap: 8px;
            margin: 14px 0 4px;
        }

        .flow-node {
            border: 1px solid var(--border);
            background: rgba(255,255,255,.035);
            border-radius: 14px;
            padding: 12px 8px;
            text-align: center;
            font-size: .82rem;
            font-weight: 750;
        }

        .flow-arrow {
            color: #8f8197;
            font-weight: 900;
        }

        .disclaimer {
            margin-top: 18px;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid rgba(185,120,255,.18);
            background: rgba(185,120,255,.06);
            color: #c8bacf;
            font-size: .82rem;
            line-height: 1.55;
        }

        div[data-testid="stCameraInput"],
        div[data-testid="stFileUploader"] {
            border-radius: 18px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 14px;
            border: 1px solid rgba(185,120,255,.28);
            background: linear-gradient(135deg, #a661e8, #7b3bb2);
            color: white;
            font-weight: 800;
        }

        div.stButton > button:hover {
            border-color: rgba(255,255,255,.35);
            color: white;
        }

        @media (max-width: 520px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }

            .flow {
                grid-template-columns: 1fr;
            }

            .flow-arrow {
                transform: rotate(90deg);
                text-align: center;
            }

            .subtitle {
                margin-left: 0;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 4. HÀM XỬ LÝ ẢNH
# -----------------------------
def clamp_roi(roi):
    x1, y1, x2, y2 = roi
    vals = [max(0.0, min(1.0, float(v))) for v in (x1, y1, x2, y2)]
    x1, y1, x2, y2 = vals

    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI không hợp lệ.")

    return x1, y1, x2, y2


def crop_fractional_roi(image: Image.Image, roi):
    x1, y1, x2, y2 = clamp_roi(roi)
    w, h = image.size

    left = int(round(x1 * w))
    top = int(round(y1 * h))
    right = int(round(x2 * w))
    bottom = int(round(y2 * h))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))

    return image.crop((left, top, right, bottom))


def median_rgb(image: Image.Image):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)

    if arr.size == 0:
        raise ValueError("Vùng ảnh rỗng.")

    values = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(round(v)) for v in values)


def apply_gray_calibration(sample_rgb, gray_rgb):
    sample = np.asarray(sample_rgb, dtype=float)
    gray = np.asarray(gray_rgb, dtype=float)

    # Tránh chia cho 0 hoặc hiệu chỉnh cực đoan.
    gray = np.clip(gray, 20.0, 245.0)

    gains = GRAY_TARGET_RGB / gray
    gains = np.clip(gains, 0.65, 1.55)

    corrected = np.clip(sample * gains, 0, 255)
    return tuple(int(round(v)) for v in corrected)


def rgb_to_hsv_features(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # H: độ, S/V: %
    return round(h * 360.0, 2), round(s * 100.0, 2), round(v * 100.0, 2)


def build_feature_vector(rgb):
    r, g, b = rgb
    h, s, v = rgb_to_hsv_features(rgb)
    return np.array([[r, g, b, h, s, v]], dtype=float), (h, s, v)


def draw_roi_overlay(image: Image.Image, indicator_roi, gray_roi=None):
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    w, h = preview.size

    def rect_from_roi(roi):
        x1, y1, x2, y2 = clamp_roi(roi)
        return (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )

    # Tím: vùng thẻ chỉ thị
    ir = rect_from_roi(indicator_roi)
    width = max(2, int(min(w, h) * 0.008))
    draw.rectangle(ir, outline=(190, 120, 255), width=width)

    # Xám: vùng thẻ tham chiếu
    if gray_roi is not None:
        gr = rect_from_roi(gray_roi)
        draw.rectangle(gr, outline=(220, 220, 220), width=width)

    return preview


# -----------------------------
# 5. MODEL
# -----------------------------
@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None

    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def normalize_model_label(raw_label):
    text = str(raw_label).strip().lower()

    mapping = {
        "fresh": "fresh",
        "còn tươi": "fresh",
        "con tuoi": "fresh",
        "0": "fresh",

        "transition": "transition",
        "chuyển tiếp": "transition",
        "chuyen tiep": "transition",
        "nên sử dụng sớm": "transition",
        "nen su dung som": "transition",
        "1": "transition",

        "spoiled": "spoiled",
        "hư hỏng": "spoiled",
        "hu hong": "spoiled",
        "có dấu hiệu hư hỏng": "spoiled",
        "co dau hieu hu hong": "spoiled",
        "2": "spoiled",
    }

    if text in mapping:
        return mapping[text]

    raise ValueError(
        f"Nhãn model '{raw_label}' chưa được ánh xạ. "
        "Hãy sửa hàm normalize_model_label()."
    )


def predict_with_model(model, feature_vector):
    raw = model.predict(feature_vector)[0]
    return normalize_model_label(raw)


def predict_demo(rgb):
    """
    Chỉ dùng để thử giao diện khi chưa có model thật.
    Không dùng kết quả demo làm dữ liệu nghiên cứu.
    """
    sample = np.asarray(rgb, dtype=float)
    distances = {
        label: float(np.linalg.norm(sample - centroid))
        for label, centroid in DEMO_CENTROIDS.items()
    }
    return min(distances, key=distances.get)


# -----------------------------
# 6. HÀM HIỂN THỊ KẾT QUẢ
# -----------------------------
def result_style(label):
    if label == "fresh":
        return {
            "css": "result-fresh",
            "icon": "🟢",
            "text": CLASS_LABELS[label],
            "desc": "Mẫu được mô hình xếp vào nhóm còn tươi trong điều kiện thí nghiệm.",
        }

    if label == "transition":
        return {
            "css": "result-transition",
            "icon": "🟡",
            "text": CLASS_LABELS[label],
            "desc": "Mẫu được mô hình xếp vào nhóm chuyển tiếp; nên ưu tiên sử dụng sớm.",
        }

    return {
        "css": "result-spoiled",
        "icon": "🔴",
        "text": CLASS_LABELS["spoiled"],
        "desc": "Mẫu được mô hình xếp vào nhóm có dấu hiệu hư hỏng.",
    }


def render_result(label, rgb, hsv, calibrated, mode_name):
    style = result_style(label)
    r, g, b = rgb
    h, s, v = hsv

    calibrated_text = "Có" if calibrated else "Không"

    st.markdown(
        f"""
        <div class="result-card {style['css']}">
            <div class="result-label">Kết quả phân loại</div>
            <div class="result-main">{style['icon']} {style['text']}</div>
            <div class="result-desc">{style['desc']}</div>

            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Màu thẻ RGB</div>
                    <div class="metric-value">
                        <span class="swatch" style="background: rgb({r},{g},{b});"></span>
                        {r}, {g}, {b}
                    </div>
                </div>

                <div class="metric">
                    <div class="metric-label">HSV</div>
                    <div class="metric-value">H {h:.1f}° · S {s:.1f}% · V {v:.1f}%</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Hiệu chỉnh thẻ xám</div>
                    <div class="metric-value">{calibrated_text}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Bộ phân loại</div>
                    <div class="metric-value">{mode_name}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 7. HEADER
# -----------------------------
st.markdown(
    """
    <div class="brand">
        <div class="brand-mark">🍃</div>
        <div class="brand-name">FreshTag</div>
    </div>
    <div class="subtitle">
        Phân loại trạng thái ức gà bằng thẻ chỉ thị chitosan–tía tô và ảnh smartphone
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 8. TABS
# -----------------------------
tab_check, tab_explain = st.tabs(["🔬 Kiểm tra mẫu", "🧠 Cách hoạt động"])


with tab_check:
    model = load_trained_model()
    is_real_model = model is not None

    if is_real_model:
        st.success("Đã tải mô hình học máy từ model.joblib.")
    else:
        st.warning(
            "Chưa có model.joblib — ứng dụng đang chạy ở CHẾ ĐỘ DEMO để thử giao diện. "
            "Không dùng kết quả demo làm kết quả nghiên cứu."
        )

    st.markdown(
        """
        <div class="card">
            <div class="section-kicker">Bước 1</div>
            <div class="section-title">Đưa ảnh thẻ chỉ thị vào ứng dụng</div>
            <div class="helper">
                Nên chụp bằng cùng một điện thoại, cùng hộp ánh sáng và cùng vị trí như quy trình thí nghiệm.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Nguồn ảnh",
        ["📷 Chụp ảnh", "⬆️ Tải ảnh lên"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None

    if input_mode == "📷 Chụp ảnh":
        uploaded_file = st.camera_input(
            "Chụp ảnh thẻ chỉ thị",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "Tải ảnh JPG/PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    use_gray_card = st.checkbox(
        "Hiệu chỉnh màu bằng thẻ tham chiếu",
        value=True,
        help="Bỏ chọn nếu ảnh không có thẻ xám ở đúng vị trí đã quy định.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Không đọc được ảnh. Hãy thử lại bằng file JPG hoặc PNG.")
            st.stop()

        st.markdown(
            """
            <div class="card">
                <div class="section-kicker">Bước 2</div>
                <div class="section-title">Kiểm tra vùng đọc màu</div>
                <div class="helper">
                    Khung tím là vùng đọc thẻ chỉ thị. Khung xám là vùng thẻ tham chiếu.
                    Hai vị trí này nên được cố định trong hộp chụp ảnh.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        overlay = draw_roi_overlay(
            image,
            INDICATOR_ROI,
            GRAY_CARD_ROI if use_gray_card else None,
        )
        st.image(overlay, use_container_width=True)

        analyze = st.button("✨ PHÂN TÍCH MÀU THẺ", type="primary")

        if analyze:
            try:
                indicator_crop = crop_fractional_roi(image, INDICATOR_ROI)
                raw_rgb = median_rgb(indicator_crop)

                final_rgb = raw_rgb
                gray_rgb = None

                if use_gray_card:
                    gray_crop = crop_fractional_roi(image, GRAY_CARD_ROI)
                    gray_rgb = median_rgb(gray_crop)
                    final_rgb = apply_gray_calibration(raw_rgb, gray_rgb)

                feature_vector, hsv = build_feature_vector(final_rgb)

                if is_real_model:
                    label = predict_with_model(model, feature_vector)
                    mode_name = "Mô hình ML"
                else:
                    label = predict_demo(final_rgb)
                    mode_name = "DEMO — chưa phải model nghiên cứu"

                render_result(
                    label=label,
                    rgb=final_rgb,
                    hsv=hsv,
                    calibrated=use_gray_card,
                    mode_name=mode_name,
                )

                with st.expander("Xem chi tiết kỹ thuật"):
                    st.write("RGB thô vùng thẻ:", raw_rgb)

                    if gray_rgb is not None:
                        st.write("RGB thẻ tham chiếu:", gray_rgb)
                        st.write("RGB sau hiệu chỉnh:", final_rgb)

                    st.write("Đặc trưng đưa vào model:", FEATURE_COLUMNS)
                    st.write(
                        {
                            "R": int(final_rgb[0]),
                            "G": int(final_rgb[1]),
                            "B": int(final_rgb[2]),
                            "H": float(hsv[0]),
                            "S": float(hsv[1]),
                            "V": float(hsv[2]),
                        }
                    )

                    st.image(indicator_crop, caption="Vùng thẻ chỉ thị được phân tích")

            except Exception as exc:
                st.error(f"Không thể phân tích ảnh: {exc}")

    else:
        st.info("Chụp ảnh hoặc tải ảnh lên để bắt đầu.")

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Lưu ý:</strong> Kết quả của ứng dụng là nhận định sàng lọc trong điều kiện
            thí nghiệm của đề tài. Ứng dụng không xác nhận thực phẩm chắc chắn an toàn để ăn
            và không thay thế kiểm nghiệm an toàn thực phẩm.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_explain:
    st.markdown(
        """
        <div class="card">
            <div class="section-kicker">Quy trình</div>
            <div class="section-title">Ứng dụng phân loại như thế nào?</div>

            <div class="flow">
                <div class="flow-node">Ảnh smartphone</div>
                <div class="flow-arrow">→</div>
                <div class="flow-node">RGB / HSV của thẻ</div>
                <div class="flow-arrow">→</div>
                <div class="flow-node">Mô hình phân loại</div>
            </div>

            <div class="helper" style="margin-top:14px;">
                Đầu ra chỉ gồm ba nhóm: <strong>Còn tươi</strong>,
                <strong>Nên sử dụng sớm</strong> và
                <strong>Có dấu hiệu hư hỏng</strong>.
                Ứng dụng không chuyển kết quả thành “% độ tươi”.
            </div>
        </div>

        <div class="card">
            <div class="section-kicker">Vùng ảnh</div>
            <div class="section-title">Tại sao phải cố định vị trí thẻ?</div>
            <div class="helper">
                Nếu vị trí camera, ánh sáng và thẻ thay đổi giữa các lần chụp,
                RGB có thể thay đổi dù mẫu không thay đổi. Vì vậy app chỉ đọc một
                vùng cố định của thẻ và có thể dùng thẻ xám để giảm sai lệch màu.
            </div>
        </div>

        <div class="card">
            <div class="section-kicker">Mô hình</div>
            <div class="section-title">Khi có dữ liệu thí nghiệm thật</div>
            <div class="helper">
                Huấn luyện Decision Tree hoặc k-NN bằng dữ liệu của chính nhóm,
                lưu model dưới tên <strong>model.joblib</strong>, rồi đặt file này
                cùng thư mục với <strong>app.py</strong>. Ứng dụng sẽ tự ưu tiên
                model thật thay cho chế độ demo.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
