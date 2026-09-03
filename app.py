import colorsys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# FreshTag — giao diện người dùng tối giản
# Flow duy nhất:
#   Chụp / tải ảnh -> tự phân tích -> hiện kết quả
#
# Khi có model thật, đặt file "model.joblib" cùng thư mục app.py.
# ============================================================

MODEL_PATH = Path("model.joblib")

# ROI cố định — cần chỉnh lại theo ảnh thật từ hộp chụp.
INDICATOR_ROI = (0.35, 0.35, 0.65, 0.65)
GRAY_CARD_ROI = (0.03, 0.03, 0.18, 0.18)
GRAY_TARGET_RGB = np.array([200.0, 200.0, 200.0], dtype=float)

# Chỉ để thử UI trước khi có model nghiên cứu thật.
# Không dùng kết quả này làm dữ liệu/bằng chứng khoa học.
DEMO_CENTROIDS = {
    "fresh": np.array([205.0, 95.0, 135.0]),
    "transition": np.array([145.0, 110.0, 165.0]),
    "spoiled": np.array([75.0, 135.0, 125.0]),
}

RESULTS = {
    "fresh": {
        "icon": "✓",
        "title": "CÒN TƯƠI",
        "message": "Thực phẩm đang ở trạng thái còn tươi.",
        "class": "fresh",
    },
    "transition": {
        "icon": "!",
        "title": "NÊN SỬ DỤNG SỚM",
        "message": "Thực phẩm đang chuyển trạng thái. Nên ưu tiên sử dụng sớm.",
        "class": "warning",
    },
    "spoiled": {
        "icon": "×",
        "title": "CÓ DẤU HIỆU HƯ HỎNG",
        "message": "Thực phẩm có dấu hiệu hư hỏng. Không nên sử dụng.",
        "class": "danger",
    },
}

st.set_page_config(
    page_title="FreshTag",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 50% -10%, rgba(133, 78, 177, .16), transparent 30%),
                #fbf9fc;
            color: #241b29;
        }

        #MainMenu, header, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 520px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }

        .brand {
            text-align: center;
            margin-bottom: 26px;
        }

        .logo {
            width: 58px;
            height: 58px;
            margin: 0 auto 12px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            color: white;
            font-size: 28px;
            background: linear-gradient(145deg, #9c62c7, #744092);
            box-shadow: 0 12px 32px rgba(116,64,146,.18);
        }

        .brand-name {
            font-size: 2rem;
            font-weight: 850;
            letter-spacing: -.05em;
            line-height: 1;
        }

        .tagline {
            margin-top: 9px;
            color: #7d7282;
            font-size: .94rem;
        }

        .intro {
            text-align: center;
            margin: 8px 0 20px;
        }

        .intro-title {
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .intro-text {
            color: #85788b;
            font-size: .9rem;
            line-height: 1.5;
        }

        div[data-testid="stCameraInput"],
        div[data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px solid #ece5ef;
            border-radius: 20px;
            padding: 6px;
            box-shadow: 0 8px 26px rgba(57,35,66,.05);
        }

        div[role="radiogroup"] {
            background: #f1edf4;
            border-radius: 14px;
            padding: 4px;
            margin-bottom: 14px;
        }

        .analyzing {
            text-align: center;
            color: #7d7282;
            font-size: .92rem;
            padding: 12px 0 4px;
        }

        .result-card {
            margin-top: 20px;
            border-radius: 26px;
            padding: 30px 22px 26px;
            text-align: center;
            border: 1px solid transparent;
            animation: enter .28s ease-out;
        }

        @keyframes enter {
            from { opacity: 0; transform: translateY(5px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .result-card.fresh {
            background: #eef9f2;
            border-color: #d5efde;
            color: #21653e;
        }

        .result-card.warning {
            background: #fff8e8;
            border-color: #f6e6ba;
            color: #805f14;
        }

        .result-card.danger {
            background: #fff0f0;
            border-color: #f4d4d4;
            color: #8a3030;
        }

        .result-icon {
            width: 68px;
            height: 68px;
            margin: 0 auto 15px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: rgba(255,255,255,.72);
            font-size: 34px;
            font-weight: 800;
        }

        .result-caption {
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            opacity: .62;
            margin-bottom: 7px;
        }

        .result-title {
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -.035em;
            line-height: 1.18;
        }

        .result-message {
            margin: 10px auto 0;
            max-width: 360px;
            font-size: .94rem;
            line-height: 1.55;
            opacity: .82;
        }

        .small-note {
            text-align: center;
            margin: 18px auto 0;
            max-width: 410px;
            color: #958a99;
            font-size: .72rem;
            line-height: 1.45;
        }

        @media (max-width: 480px) {
            .block-container {
                padding-top: 1.3rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .brand-name {
                font-size: 1.75rem;
            }

            .result-card {
                padding: 26px 17px 22px;
            }

            .result-title {
                font-size: 1.35rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def crop_roi(image, roi):
    w, h = image.size
    x1, y1, x2, y2 = roi
    return image.crop((
        int(x1 * w),
        int(y1 * h),
        int(x2 * w),
        int(y2 * h),
    ))


def median_rgb(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(round(v)) for v in rgb)


def calibrate_rgb(sample_rgb, gray_rgb):
    sample = np.asarray(sample_rgb, dtype=float)
    gray = np.clip(np.asarray(gray_rgb, dtype=float), 20.0, 245.0)
    gains = np.clip(GRAY_TARGET_RGB / gray, 0.65, 1.55)
    corrected = np.clip(sample * gains, 0, 255)
    return tuple(int(round(v)) for v in corrected)


def make_features(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return np.array([[
        r, g, b,
        h * 360,
        s * 100,
        v * 100,
    ]], dtype=float)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def normalize_label(label):
    t = str(label).strip().lower()

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

    return mapping.get(t)


def demo_predict(rgb):
    arr = np.asarray(rgb, dtype=float)
    return min(
        DEMO_CENTROIDS,
        key=lambda k: np.linalg.norm(arr - DEMO_CENTROIDS[k]),
    )


def analyze(image):
    sample_crop = crop_roi(image, INDICATOR_ROI)
    gray_crop = crop_roi(image, GRAY_CARD_ROI)

    sample_rgb = median_rgb(sample_crop)
    gray_rgb = median_rgb(gray_crop)
    final_rgb = calibrate_rgb(sample_rgb, gray_rgb)

    model = load_model()

    if model is not None:
        raw_label = model.predict(make_features(final_rgb))[0]
        label = normalize_label(raw_label)
        if label is None:
            raise ValueError("Nhãn đầu ra của mô hình chưa đúng định dạng.")
        return label

    # Chỉ phục vụ chạy thử giao diện trong giai đoạn chưa có model thật.
    return demo_predict(final_rgb)


def show_result(label):
    result = RESULTS[label]

    st.markdown(
        f"""
        <div class="result-card {result['class']}" role="status" aria-live="polite">
            <div class="result-icon">{result['icon']}</div>
            <div class="result-caption">Kết quả</div>
            <div class="result-title">{result['title']}</div>
            <div class="result-message">{result['message']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------ UI ------------------

st.markdown(
    """
    <div class="brand">
        <div class="logo">🍃</div>
        <div class="brand-name">FreshTag</div>
        <div class="tagline">Kiểm tra trạng thái thực phẩm bằng thẻ chỉ thị màu</div>
    </div>

    <div class="intro">
        <div class="intro-title">Chụp thẻ chỉ thị để kiểm tra</div>
        <div class="intro-text">
            Đặt thẻ đúng vị trí trong khung chụp, sau đó chụp ảnh hoặc chọn ảnh có sẵn.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Chọn cách nhập ảnh",
    ["📷 Chụp ảnh", "🖼️ Chọn từ thư viện"],
    horizontal=True,
    label_visibility="collapsed",
)

image_file = None

if mode == "📷 Chụp ảnh":
    image_file = st.camera_input(
        "Chụp ảnh",
        label_visibility="collapsed",
    )
else:
    image_file = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")

        # Không cần nút "Phân tích": có ảnh là xử lý ngay.
        label = analyze(image)
        show_result(label)

    except Exception:
        st.error("Không thể đọc ảnh. Vui lòng chụp lại hoặc chọn ảnh khác.")

st.markdown(
    """
    <div class="small-note">
        Kết quả chỉ mang tính hỗ trợ nhận định và không thay thế kiểm nghiệm an toàn thực phẩm.
    </div>
    """,
    unsafe_allow_html=True,
)
