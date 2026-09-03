import colorsys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# FreshTag — Cinematic UI
# Flow duy nhất: Chụp / tải ảnh -> tự phân tích -> hiện kết quả
#
# Khi có model thật, đặt "model.joblib" cùng thư mục app.py.
# ============================================================

MODEL_PATH = Path("model.joblib")

# ROI tạm thời. Sau khi chốt hộp chụp/camera, chỉnh lại đúng vị trí thẻ.
INDICATOR_ROI = (0.35, 0.35, 0.65, 0.65)

# Demo centroids chỉ để app chạy trước khi có model thật.
DEMO_CENTROIDS = {
    "fresh": np.array([205.0, 95.0, 135.0]),
    "transition": np.array([145.0, 110.0, 165.0]),
    "spoiled": np.array([75.0, 135.0, 125.0]),
}

RESULTS = {
    "fresh": {
        "title": "CÒN TƯƠI",
        "message": "Thực phẩm đang ở trạng thái còn tươi.",
        "class": "fresh",
        "symbol": "✓",
    },
    "transition": {
        "title": "NÊN SỬ DỤNG SỚM",
        "message": "Thực phẩm đang chuyển trạng thái. Nên ưu tiên sử dụng sớm.",
        "class": "warning",
        "symbol": "!",
    },
    "spoiled": {
        "title": "CÓ DẤU HIỆU HƯ HỎNG",
        "message": "Thực phẩm có dấu hiệu hư hỏng. Không nên sử dụng.",
        "class": "danger",
        "symbol": "!",
    },
}


def _html(s: str) -> str:
    return "\n".join(line.strip() for line in s.strip().split("\n"))


# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------
st.set_page_config(
    page_title="FreshTag",
    page_icon="🟣",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# BASE CSS — cinematic dark plum
# ------------------------------------------------------------
st.markdown(
    _html(r"""
    <style>
    :root{
        --bg0:#09070d;
        --bg1:#120b18;
        --bg2:#1b1024;
        --ink:#f8f4fb;
        --muted:#b8aebe;
        --purple:#a860d1;
        --purple2:#7f3fa7;
        --magenta:#c65c9f;
        --line:rgba(255,255,255,.10);
        --glass:rgba(255,255,255,.055);
    }

    html, body, [class*="css"]{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body{
        background:#09070d;
    }

    .stApp{
        color:var(--ink);
        background:
            radial-gradient(circle at 14% 4%, rgba(155,76,194,.18), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(190,64,137,.12), transparent 26%),
            radial-gradient(circle at 68% 92%, rgba(92,40,123,.13), transparent 30%),
            linear-gradient(145deg,var(--bg0) 0%,var(--bg1) 48%,#0c0811 100%);
        min-height:100vh;
        transition:background .55s ease;
    }

    #MainMenu, header, footer{
        visibility:hidden;
    }

    .block-container{
        max-width:760px;
        padding-top:1.15rem;
        padding-bottom:2.4rem;
        position:relative;
        z-index:2;
    }

    /* =======================================================
       BACKGROUND PERILLA LINE ART
       ======================================================= */
    .botanical-layer{
        position:fixed;
        inset:0;
        z-index:0;
        pointer-events:none;
        overflow:hidden;
    }

    .leaf-art{
        position:absolute;
        width:360px;
        height:360px;
        opacity:.075;
        filter:drop-shadow(0 0 20px rgba(180,92,210,.14));
    }

    .leaf-art.left{
        left:-105px;
        top:110px;
        transform:rotate(-18deg);
    }

    .leaf-art.right{
        right:-115px;
        bottom:30px;
        transform:rotate(168deg) scale(.92);
    }

    .leaf-line{
        fill:none;
        stroke:#d6a3ee;
        stroke-width:2.2;
        stroke-linecap:round;
        stroke-linejoin:round;
    }

    /* =======================================================
       HERO
       ======================================================= */
    .hero{
        position:relative;
        overflow:hidden;
        border-radius:30px;
        padding:30px 32px 29px;
        margin-bottom:18px;
        background:
            radial-gradient(circle at 82% 8%, rgba(255,255,255,.10), transparent 21%),
            radial-gradient(circle at 16% 96%, rgba(208,91,167,.13), transparent 27%),
            linear-gradient(135deg, rgba(86,38,113,.78), rgba(48,22,66,.88) 48%, rgba(76,25,68,.80));
        border:1px solid rgba(255,255,255,.10);
        box-shadow:
            0 24px 70px rgba(0,0,0,.35),
            inset 0 1px 0 rgba(255,255,255,.08);
        backdrop-filter:blur(16px);
    }

    .hero:before{
        content:"";
        position:absolute;
        inset:-1px;
        border-radius:30px;
        padding:1px;
        background:linear-gradient(130deg,rgba(225,163,255,.35),rgba(255,255,255,.02),rgba(220,95,163,.22));
        -webkit-mask:
            linear-gradient(#fff 0 0) content-box,
            linear-gradient(#fff 0 0);
        -webkit-mask-composite:xor;
        mask-composite:exclude;
        pointer-events:none;
    }

    .hero-orb{
        position:absolute;
        right:-70px;
        top:-90px;
        width:230px;
        height:230px;
        border-radius:50%;
        background:radial-gradient(circle,rgba(202,105,224,.22),rgba(202,105,224,0) 68%);
        filter:blur(2px);
    }

    .brand-row{
        display:flex;
        align-items:center;
        gap:13px;
        position:relative;
        z-index:1;
    }

    .brand-mark{
        width:52px;
        height:52px;
        display:grid;
        place-items:center;
        border-radius:16px;
        background:linear-gradient(145deg,rgba(188,101,215,.28),rgba(123,61,160,.16));
        border:1px solid rgba(236,194,255,.20);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.10),
            0 10px 30px rgba(0,0,0,.22);
    }

    .brand-mark svg{
        width:33px;
        height:33px;
    }

    .brand-name{
        font-size:1.82rem;
        font-weight:900;
        letter-spacing:-.055em;
        line-height:1;
    }

    .brand-mini{
        margin-top:5px;
        font-size:.71rem;
        font-weight:700;
        letter-spacing:.16em;
        color:rgba(255,255,255,.55);
        text-transform:uppercase;
    }

    .hero-copy{
        position:relative;
        z-index:1;
        margin-top:29px;
        max-width:540px;
    }

    .hero-kicker{
        display:inline-flex;
        align-items:center;
        gap:7px;
        padding:6px 9px;
        border-radius:999px;
        font-size:.66rem;
        font-weight:800;
        letter-spacing:.11em;
        text-transform:uppercase;
        color:#dbc8e3;
        background:rgba(255,255,255,.055);
        border:1px solid rgba(255,255,255,.08);
        margin-bottom:12px;
    }

    .hero-title{
        font-size:2.15rem;
        line-height:1.04;
        font-weight:920;
        letter-spacing:-.055em;
        margin:0;
        max-width:530px;
    }

    .hero-sub{
        margin-top:12px;
        color:rgba(255,255,255,.64);
        font-size:.94rem;
        line-height:1.58;
        max-width:470px;
    }

    /* =======================================================
       ACTION / INPUT
       ======================================================= */
    .action-head{
        text-align:center;
        margin:18px 0 12px;
    }

    .action-title{
        font-size:1.1rem;
        font-weight:850;
        letter-spacing:-.025em;
    }

    .action-sub{
        margin-top:4px;
        color:#9f93a5;
        font-size:.84rem;
    }

    div[role="radiogroup"]{
        width:fit-content;
        margin:0 auto 13px auto;
        padding:5px 7px;
        border-radius:16px;
        background:rgba(255,255,255,.045);
        border:1px solid rgba(255,255,255,.08);
        box-shadow:0 12px 32px rgba(0,0,0,.16);
        backdrop-filter:blur(16px);
    }

    div[role="radiogroup"] label{
        color:#ddd3e1 !important;
    }

    div[data-testid="stCameraInput"],
    div[data-testid="stFileUploader"]{
        background:rgba(255,255,255,.055);
        border:1px solid rgba(255,255,255,.09);
        border-radius:26px;
        padding:8px;
        overflow:hidden;
        box-shadow:
            0 24px 60px rgba(0,0,0,.24),
            inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter:blur(18px);
    }

    div[data-testid="stCameraInput"] video{
        border-radius:20px !important;
        max-height:420px !important;
        object-fit:cover !important;
        background:#050407 !important;
    }

    div[data-testid="stCameraInput"] button,
    div[data-testid="stFileUploader"] button{
        border-radius:14px !important;
        font-weight:800 !important;
    }

    /* =======================================================
       RESULT — default dark glass
       ======================================================= */
    .result-shell{
        position:relative;
        margin-top:18px;
        padding:1px;
        border-radius:29px;
        overflow:hidden;
        animation:resultIn .38s cubic-bezier(.2,.8,.2,1);
    }

    @keyframes resultIn{
        from{opacity:0; transform:translateY(8px) scale(.992);}
        to{opacity:1; transform:translateY(0) scale(1);}
    }

    .result-card{
        position:relative;
        z-index:1;
        border-radius:28px;
        padding:30px 25px 28px;
        text-align:center;
        backdrop-filter:blur(20px);
        box-shadow:0 24px 70px rgba(0,0,0,.30);
    }

    .status-mark{
        width:78px;
        height:78px;
        margin:0 auto 16px;
        display:grid;
        place-items:center;
        border-radius:24px;
        font-size:38px;
        line-height:1;
        font-weight:900;
    }

    .status-caption{
        font-size:.67rem;
        font-weight:850;
        letter-spacing:.18em;
        text-transform:uppercase;
        opacity:.58;
        margin-bottom:7px;
    }

    .status-title{
        font-size:1.62rem;
        line-height:1.13;
        font-weight:930;
        letter-spacing:-.04em;
    }

    .status-message{
        max-width:430px;
        margin:10px auto 0;
        font-size:.92rem;
        line-height:1.58;
        opacity:.72;
    }

    /* Fresh */
    .result-shell.fresh{
        background:linear-gradient(135deg,rgba(73,220,145,.62),rgba(116,76,166,.22));
        box-shadow:0 0 54px rgba(47,194,117,.14);
    }
    .fresh .result-card{
        background:linear-gradient(155deg,rgba(11,34,26,.95),rgba(12,20,21,.93));
        border:1px solid rgba(95,232,161,.14);
    }
    .fresh .status-mark{
        color:#66e6a5;
        background:rgba(70,213,139,.11);
        border:1px solid rgba(100,232,169,.20);
        box-shadow:0 0 34px rgba(61,207,132,.12);
    }
    .fresh .status-title{color:#7cedb4;}

    /* Warning */
    .result-shell.warning{
        background:linear-gradient(135deg,rgba(255,181,74,.70),rgba(175,95,63,.28));
        box-shadow:0 0 60px rgba(255,167,54,.17);
    }
    .warning .result-card{
        background:linear-gradient(155deg,rgba(41,28,12,.96),rgba(22,17,13,.94));
        border:1px solid rgba(255,194,95,.16);
    }
    .warning .status-mark{
        color:#ffc25c;
        background:rgba(255,176,55,.11);
        border:1px solid rgba(255,194,95,.22);
        box-shadow:0 0 34px rgba(255,173,54,.14);
    }
    .warning .status-title{color:#ffd078;}

    /* Danger */
    .result-shell.danger{
        background:linear-gradient(135deg,rgba(255,77,92,.90),rgba(136,27,50,.50));
        box-shadow:
            0 0 72px rgba(255,47,75,.24),
            0 26px 80px rgba(0,0,0,.38);
    }
    .danger .result-card{
        background:
            radial-gradient(circle at 50% -10%,rgba(193,42,65,.20),transparent 34%),
            linear-gradient(155deg,rgba(41,9,16,.98),rgba(19,7,11,.97));
        border:1px solid rgba(255,96,110,.20);
    }
    .danger .status-mark{
        color:#ff6976;
        background:rgba(255,75,91,.12);
        border:1px solid rgba(255,104,118,.26);
        box-shadow:
            0 0 42px rgba(255,60,80,.22),
            inset 0 0 24px rgba(255,71,87,.05);
    }
    .danger .status-title{
        color:#ff7d88;
        text-shadow:0 0 22px rgba(255,71,91,.13);
    }

    .note{
        text-align:center;
        max-width:540px;
        margin:15px auto 0;
        color:#776c7d;
        font-size:.68rem;
        line-height:1.45;
    }

    /* =======================================================
       RESPONSIVE
       ======================================================= */
    @media (max-width: 620px){
        .block-container{
            padding-top:.65rem;
            padding-left:.85rem;
            padding-right:.85rem;
        }

        .hero{
            border-radius:25px;
            padding:23px 20px 23px;
        }

        .hero-title{
            font-size:1.73rem;
        }

        .hero-sub{
            font-size:.88rem;
        }

        .brand-mark{
            width:47px;
            height:47px;
            border-radius:15px;
        }

        .brand-mark svg{
            width:30px;
            height:30px;
        }

        .brand-name{
            font-size:1.58rem;
        }

        .leaf-art{
            width:260px;
            height:260px;
            opacity:.052;
        }

        .leaf-art.left{
            left:-105px;
            top:165px;
        }

        .leaf-art.right{
            right:-100px;
            bottom:60px;
        }

        .result-card{
            padding:26px 17px 24px;
        }

        .status-title{
            font-size:1.38rem;
        }

        div[role="radiogroup"]{
            width:100%;
        }
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# BOTANICAL BACKGROUND — perilla-inspired leaf line art
# ------------------------------------------------------------
st.markdown(
    _html("""
    <div class="botanical-layer" aria-hidden="true">
        <svg class="leaf-art left" viewBox="0 0 320 320">
            <path class="leaf-line" d="M164 285 C147 240 130 198 124 159 C116 111 131 69 169 38 C204 72 224 113 217 154 C211 193 187 239 164 285 Z"/>
            <path class="leaf-line" d="M164 285 C165 240 164 194 166 147 C168 105 169 70 169 38"/>
            <path class="leaf-line" d="M164 230 C139 215 121 198 105 178"/>
            <path class="leaf-line" d="M165 208 C192 192 208 175 225 153"/>
            <path class="leaf-line" d="M164 178 C141 162 124 146 111 129"/>
            <path class="leaf-line" d="M166 155 C188 140 205 121 216 102"/>
            <path class="leaf-line" d="M165 127 C144 112 132 98 124 85"/>
            <path class="leaf-line" d="M167 105 C185 91 197 77 204 65"/>
            <path class="leaf-line" d="M124 159 C111 151 102 141 97 130 C108 126 115 121 119 111 C107 105 101 96 101 87 C115 87 124 83 130 74 C119 67 116 58 118 49 C130 53 142 50 151 42"/>
            <path class="leaf-line" d="M217 154 C230 144 237 134 240 123 C229 120 223 114 219 104 C232 98 238 89 238 80 C225 80 216 76 210 68 C220 60 223 52 221 44 C210 47 198 44 188 38"/>
        </svg>

        <svg class="leaf-art right" viewBox="0 0 320 320">
            <path class="leaf-line" d="M164 285 C147 240 130 198 124 159 C116 111 131 69 169 38 C204 72 224 113 217 154 C211 193 187 239 164 285 Z"/>
            <path class="leaf-line" d="M164 285 C165 240 164 194 166 147 C168 105 169 70 169 38"/>
            <path class="leaf-line" d="M164 230 C139 215 121 198 105 178"/>
            <path class="leaf-line" d="M165 208 C192 192 208 175 225 153"/>
            <path class="leaf-line" d="M164 178 C141 162 124 146 111 129"/>
            <path class="leaf-line" d="M166 155 C188 140 205 121 216 102"/>
            <path class="leaf-line" d="M165 127 C144 112 132 98 124 85"/>
            <path class="leaf-line" d="M167 105 C185 91 197 77 204 65"/>
            <path class="leaf-line" d="M124 159 C111 151 102 141 97 130 C108 126 115 121 119 111 C107 105 101 96 101 87 C115 87 124 83 130 74 C119 67 116 58 118 49 C130 53 142 50 151 42"/>
            <path class="leaf-line" d="M217 154 C230 144 237 134 240 123 C229 120 223 114 219 104 C232 98 238 89 238 80 C225 80 216 76 210 68 C220 60 223 52 221 44 C210 47 198 44 188 38"/>
        </svg>
    </div>
    """),
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


def apply_result_environment(label):
    """
    Đổi mood của toàn bộ background theo trạng thái.
    Đây chỉ là thay đổi UI, không đổi logic phân loại.
    """
    if label == "fresh":
        bg = """
        radial-gradient(circle at 50% 18%, rgba(38,184,111,.14), transparent 30%),
        radial-gradient(circle at 12% 5%, rgba(112,75,170,.18), transparent 28%),
        radial-gradient(circle at 90% 80%, rgba(29,127,85,.10), transparent 30%),
        linear-gradient(145deg,#070c0a 0%,#0a1712 48%,#09090d 100%)
        """
        leaf = "#83d7ad"
        opacity = ".065"

    elif label == "transition":
        bg = """
        radial-gradient(circle at 50% 18%, rgba(225,147,42,.16), transparent 30%),
        radial-gradient(circle at 12% 5%, rgba(112,75,170,.17), transparent 28%),
        radial-gradient(circle at 88% 84%, rgba(151,86,27,.11), transparent 30%),
        linear-gradient(145deg,#0d0a06 0%,#1a1209 48%,#0d090d 100%)
        """
        leaf = "#e0b36c"
        opacity = ".068"

    else:
        bg = """
        radial-gradient(circle at 50% 20%, rgba(194,31,54,.22), transparent 30%),
        radial-gradient(circle at 12% 4%, rgba(121,31,64,.22), transparent 28%),
        radial-gradient(circle at 90% 82%, rgba(178,18,47,.15), transparent 32%),
        linear-gradient(145deg,#0a0507 0%,#1c080e 47%,#090609 100%)
        """
        leaf = "#e15f74"
        opacity = ".075"

    st.markdown(
        _html(f"""
        <style>
        .stApp{{
            background:{bg};
        }}
        .leaf-line{{
            stroke:{leaf};
        }}
        .leaf-art{{
            opacity:{opacity};
        }}
        </style>
        """),
        unsafe_allow_html=True,
    )


def show_result(label):
    r = RESULTS[label]
    st.markdown(
        _html(f"""
        <div class="result-shell {r['class']}">
            <div class="result-card">
                <div class="status-mark">{r['symbol']}</div>
                <div class="status-caption">Trạng thái</div>
                <div class="status-title">{r['title']}</div>
                <div class="status-message">{r['message']}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
st.markdown(
    _html("""
    <div class="hero">
        <div class="hero-orb"></div>

        <div class="brand-row">
            <div class="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 64 64">
                    <path d="M33 54 C27 44 21 34 20 24 C19 14 24 7 34 4 C44 10 49 18 47 28 C45 38 39 47 33 54 Z"
                          fill="none" stroke="#F0D8FA" stroke-width="2.5"
                          stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M33 54 C33 43 33 32 34 21 C34 14 34 8 34 4"
                          fill="none" stroke="#F0D8FA" stroke-width="2.2"
                          stroke-linecap="round"/>
                    <path d="M33 41 C27 37 23 33 19 28 M34 35 C40 32 44 28 48 23 M34 27 C29 24 25 20 22 16 M34 21 C39 18 42 15 45 11"
                          fill="none" stroke="#F0D8FA" stroke-width="1.8"
                          stroke-linecap="round"/>
                    <path d="M20 24 C16 21 14 18 14 15 C18 15 21 14 23 11 C20 9 19 7 20 5 C24 7 28 6 31 4
                             M47 28 C51 25 53 22 53 19 C49 19 47 18 45 15 C48 13 49 10 48 8 C45 9 42 8 39 6"
                          fill="none" stroke="#F0D8FA" stroke-width="1.7"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>

            <div>
                <div class="brand-name">FreshTag</div>
                <div class="brand-mini">Freshness indicator</div>
            </div>
        </div>

        <div class="hero-copy">
            <div class="hero-kicker">THẺ CHỈ THỊ TÍA TÔ</div>
            <div class="hero-title">Chụp một lần.<br>Biết trạng thái ngay.</div>
            <div class="hero-sub">
                Kiểm tra nhanh trạng thái thực phẩm bằng màu của thẻ chỉ thị.
            </div>
        </div>
    </div>

    <div class="action-head">
        <div class="action-title">Kiểm tra ngay</div>
        <div class="action-sub">Chụp thẻ chỉ thị hoặc chọn ảnh có sẵn</div>
    </div>
    """),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
mode = st.radio(
    "Nguồn ảnh",
    ["📷 Chụp ảnh", "▣ Thư viện"],
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

# ------------------------------------------------------------
# AUTO RESULT
# ------------------------------------------------------------
if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        label = analyze(image)

        # Thay background theo trạng thái rồi hiển thị kết quả.
        apply_result_environment(label)
        show_result(label)

    except Exception:
        st.error("Không thể đọc ảnh. Vui lòng chụp lại hoặc chọn ảnh khác.")

st.markdown(
    _html("""
    <div class="note">
        Kết quả mang tính hỗ trợ nhận định và không thay thế kiểm nghiệm an toàn thực phẩm.
    </div>
    """),
    unsafe_allow_html=True,
)
