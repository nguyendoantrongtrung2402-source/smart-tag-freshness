import colorsys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# FreshTag — UI consumer-first
# Flow:
#   Chụp / tải ảnh -> tự phân tích -> hiện kết quả
#
# Khi có model thật, đặt "model.joblib" cùng thư mục app.py.
# ============================================================

MODEL_PATH = Path("model.joblib")

# ROI tạm thời. Chỉnh lại khi chốt vị trí camera/thẻ trong hộp chụp.
INDICATOR_ROI = (0.35, 0.35, 0.65, 0.65)

# Demo centroids chỉ để app hoạt động trước khi có dữ liệu/model thật.
DEMO_CENTROIDS = {
    "fresh": np.array([205.0, 95.0, 135.0]),
    "transition": np.array([145.0, 110.0, 165.0]),
    "spoiled": np.array([75.0, 135.0, 125.0]),
}

RESULTS = {
    "fresh": {
        "emoji": "✓",
        "eyebrow": "TRẠNG THÁI",
        "title": "CÒN TƯƠI",
        "message": "Có thể tiếp tục bảo quản và sử dụng theo điều kiện phù hợp.",
        "class": "fresh",
    },
    "transition": {
        "emoji": "!",
        "eyebrow": "TRẠNG THÁI",
        "title": "NÊN SỬ DỤNG SỚM",
        "message": "Thực phẩm đang chuyển trạng thái. Nên ưu tiên sử dụng sớm.",
        "class": "warning",
    },
    "spoiled": {
        "emoji": "×",
        "eyebrow": "TRẠNG THÁI",
        "title": "CÓ DẤU HIỆU HƯ HỎNG",
        "message": "Thực phẩm có dấu hiệu hư hỏng. Không nên sử dụng.",
        "class": "danger",
    },
}

# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------
st.set_page_config(
    page_title="FreshTag",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root{
        --ink:#24162c;
        --muted:#796b80;
        --violet:#7b3fc6;
        --violet2:#b250c9;
        --pink:#ee72ad;
        --cream:#fffafd;
        --line:rgba(91,55,108,.12);
    }

    html, body, [class*="css"]{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp{
        color:var(--ink);
        background:
            radial-gradient(circle at 9% 5%, rgba(230,121,198,.22), transparent 24%),
            radial-gradient(circle at 92% 10%, rgba(127,75,202,.22), transparent 28%),
            radial-gradient(circle at 80% 88%, rgba(255,190,103,.14), transparent 28%),
            linear-gradient(145deg,#fffafd 0%,#f8f0fb 48%,#fff8fb 100%);
        min-height:100vh;
    }

    #MainMenu, header, footer{ visibility:hidden; }

    .block-container{
        max-width:720px;
        padding-top:1.2rem;
        padding-bottom:2rem;
    }

    /* ---------- HERO ---------- */
    .hero{
        position:relative;
        overflow:hidden;
        border-radius:30px;
        padding:28px 30px 26px;
        margin-bottom:16px;
        color:white;
        background:
            radial-gradient(circle at 86% 14%, rgba(255,255,255,.20), transparent 22%),
            radial-gradient(circle at 12% 90%, rgba(255,186,221,.22), transparent 24%),
            linear-gradient(135deg,#57277f 0%,#8140b0 46%,#bd4f9b 100%);
        box-shadow:0 18px 48px rgba(96,47,119,.20);
    }

    .hero:before{
        content:"";
        position:absolute;
        width:160px;
        height:160px;
        right:-55px;
        bottom:-70px;
        border:1px solid rgba(255,255,255,.15);
        border-radius:50%;
    }

    .hero:after{
        content:"";
        position:absolute;
        width:95px;
        height:95px;
        right:28px;
        top:-45px;
        border:1px solid rgba(255,255,255,.15);
        border-radius:50%;
    }

    .brand-row{
        display:flex;
        align-items:center;
        gap:12px;
        position:relative;
        z-index:1;
    }

    .brand-icon{
        width:48px;
        height:48px;
        display:grid;
        place-items:center;
        border-radius:16px;
        font-size:24px;
        background:rgba(255,255,255,.16);
        border:1px solid rgba(255,255,255,.20);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.18);
    }

    .brand-name{
        font-size:1.75rem;
        font-weight:900;
        letter-spacing:-.05em;
        line-height:1;
    }

    .brand-mini{
        margin-top:4px;
        font-size:.76rem;
        font-weight:700;
        letter-spacing:.04em;
        text-transform:uppercase;
        color:rgba(255,255,255,.70);
    }

    .hero-copy{
        position:relative;
        z-index:1;
        margin-top:24px;
        max-width:500px;
    }

    .hero-title{
        font-size:2rem;
        line-height:1.08;
        font-weight:900;
        letter-spacing:-.05em;
        margin:0;
    }

    .hero-sub{
        margin-top:10px;
        color:rgba(255,255,255,.78);
        font-size:.95rem;
        line-height:1.55;
        max-width:450px;
    }

    /* ---------- ACTION CARD ---------- */
    .action-head{
        margin:18px 0 10px;
        text-align:center;
    }

    .action-title{
        font-size:1.15rem;
        font-weight:850;
        letter-spacing:-.02em;
    }

    .action-sub{
        margin-top:4px;
        color:var(--muted);
        font-size:.86rem;
    }

    div[role="radiogroup"]{
        width:fit-content;
        margin:0 auto 12px auto;
        background:rgba(255,255,255,.68);
        border:1px solid var(--line);
        border-radius:16px;
        padding:5px 7px;
        box-shadow:0 8px 24px rgba(78,45,91,.06);
        backdrop-filter:blur(12px);
    }

    div[data-testid="stCameraInput"],
    div[data-testid="stFileUploader"]{
        background:rgba(255,255,255,.78);
        border:1px solid rgba(98,55,116,.13);
        border-radius:24px;
        padding:8px;
        box-shadow:0 16px 42px rgba(78,45,91,.09);
        overflow:hidden;
        backdrop-filter:blur(14px);
    }

    div[data-testid="stCameraInput"] video{
        border-radius:18px !important;
        max-height:390px !important;
        object-fit:cover !important;
    }

    div[data-testid="stCameraInput"] button,
    div[data-testid="stFileUploader"] button{
        border-radius:14px !important;
        font-weight:800 !important;
    }

    /* ---------- RESULT ---------- */
    .result-wrap{
        margin-top:16px;
        border-radius:28px;
        padding:6px;
        background:linear-gradient(135deg,rgba(125,63,198,.28),rgba(238,114,173,.28));
        box-shadow:0 18px 42px rgba(88,45,105,.12);
        animation:pop .28s ease-out;
    }

    .result-card{
        text-align:center;
        border-radius:23px;
        padding:28px 24px 26px;
        background:white;
    }

    @keyframes pop{
        from{opacity:0; transform:translateY(7px) scale(.99);}
        to{opacity:1; transform:translateY(0) scale(1);}
    }

    .result-icon{
        width:72px;
        height:72px;
        margin:0 auto 14px;
        display:grid;
        place-items:center;
        border-radius:24px;
        font-size:36px;
        font-weight:900;
    }

    .result-eyebrow{
        font-size:.72rem;
        font-weight:850;
        letter-spacing:.12em;
        opacity:.55;
        margin-bottom:6px;
    }

    .result-title{
        font-size:1.6rem;
        font-weight:950;
        letter-spacing:-.04em;
        line-height:1.14;
    }

    .result-message{
        margin:9px auto 0;
        max-width:430px;
        font-size:.92rem;
        line-height:1.55;
        opacity:.76;
    }

    .fresh .result-icon{
        color:#17764a;
        background:#e7f7ee;
    }
    .fresh .result-title{color:#17764a;}

    .warning .result-icon{
        color:#94630b;
        background:#fff2cf;
    }
    .warning .result-title{color:#94630b;}

    .danger .result-icon{
        color:#a03445;
        background:#ffe8ec;
    }
    .danger .result-title{color:#a03445;}

    .note{
        text-align:center;
        max-width:520px;
        margin:14px auto 0;
        color:#95899a;
        font-size:.69rem;
        line-height:1.45;
    }

    /* Compact desktop feel */
    @media (min-width: 760px){
        .block-container{ padding-top:1rem; }
        .hero{ padding:26px 32px 24px; }
        .hero-copy{ margin-top:20px; }
    }

    /* Mobile */
    @media (max-width: 560px){
        .block-container{
            padding-top:.65rem;
            padding-left:.85rem;
            padding-right:.85rem;
        }

        .hero{
            border-radius:24px;
            padding:22px 20px 21px;
        }

        .brand-icon{
            width:44px;
            height:44px;
            border-radius:14px;
        }

        .brand-name{ font-size:1.55rem; }
        .hero-title{ font-size:1.62rem; }
        .hero-sub{ font-size:.89rem; }

        .result-card{
            padding:25px 17px 22px;
        }

        .result-title{
            font-size:1.38rem;
        }

        div[role="radiogroup"]{
            width:100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# IMAGE + MODEL
# ------------------------------------------------------------
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


def make_features(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return np.array([[r, g, b, h * 360, s * 100, v * 100]], dtype=float)


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
        key=lambda key: np.linalg.norm(arr - DEMO_CENTROIDS[key]),
    )


def analyze(image):
    crop = crop_roi(image, INDICATOR_ROI)
    rgb = median_rgb(crop)
    model = load_model()

    if model is not None:
        raw = model.predict(make_features(rgb))[0]
        label = normalize_label(raw)
        if label is None:
            raise ValueError("Nhãn mô hình không hợp lệ.")
        return label

    return demo_predict(rgb)


def show_result(label):
    r = RESULTS[label]
    st.markdown(
        f"""
        <div class="result-wrap">
            <div class="result-card {r['class']}" role="status" aria-live="polite">
                <div class="result-icon">{r['emoji']}</div>
                <div class="result-eyebrow">{r['eyebrow']}</div>
                <div class="result-title">{r['title']}</div>
                <div class="result-message">{r['message']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="brand-row">
            <div class="brand-icon">🍃</div>
            <div>
                <div class="brand-name">FreshTag</div>
                <div class="brand-mini">Smart freshness indicator</div>
            </div>
        </div>

        <div class="hero-copy">
            <div class="hero-title">Chụp một lần.<br>Biết trạng thái ngay.</div>
            <div class="hero-sub">
                Dùng thẻ chỉ thị màu để nhận biết nhanh trạng thái thực phẩm.
                Đơn giản, trực quan và dễ sử dụng.
            </div>
        </div>
    </div>

    <div class="action-head">
        <div class="action-title">Kiểm tra ngay</div>
        <div class="action-sub">Chụp thẻ chỉ thị hoặc chọn ảnh có sẵn</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Nguồn ảnh",
    ["📷 Chụp ảnh", "🖼️ Thư viện"],
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
        label = analyze(image)
        show_result(label)
    except Exception:
        st.error("Không thể đọc ảnh. Vui lòng chụp lại hoặc chọn ảnh khác.")

st.markdown(
    """
    <div class="note">
        Kết quả mang tính hỗ trợ nhận định và không thay thế kiểm nghiệm an toàn thực phẩm.
    </div>
    """,
    unsafe_allow_html=True,
)
