import colorsys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = Path("model.joblib")
INDICATOR_ROI = (0.35, 0.35, 0.65, 0.65)

DEMO_CENTROIDS = {
    "fresh": np.array([205.0, 95.0, 135.0]),
    "transition": np.array([145.0, 110.0, 165.0]),
    "spoiled": np.array([75.0, 135.0, 125.0]),
}

RESULTS = {
    "fresh": {"title": "CÒN TƯƠI", "message": "Thực phẩm đang ở trạng thái còn tươi.", "class": "fresh", "symbol": "✓"},
    "transition": {"title": "NÊN SỬ DỤNG SỚM", "message": "Thực phẩm đang chuyển trạng thái. Nên ưu tiên sử dụng sớm.", "class": "warning", "symbol": "!"},
    "spoiled": {"title": "CÓ DẤU HIỆU HƯ HỎNG", "message": "Thực phẩm có dấu hiệu hư hỏng. Không nên sử dụng.", "class": "danger", "symbol": "!"},
}

def _html(s: str) -> str:
    return "\n".join(line.strip() for line in s.strip().split("\n"))

PERILLA_CLUSTER_SVG = r"""
<svg viewBox="0 0 360 300" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
    <path d="M188 151 C167 130 147 112 126 99 L111 90 L119 82 L100 78 L109 69 L88 62 L102 53 L84 44 L105 41 L96 28 L119 31 L116 17 L140 27 L145 11 L164 29 L173 18 L181 41 C190 72 196 110 188 151 Z" stroke-width="5"/>
    <path d="M181 41 C164 55 151 70 139 88 C128 104 117 120 106 139" stroke-width="3"/>
    <path d="M173 55 L151 50 M166 66 L141 64 M157 79 L131 78 M148 92 L122 93 M139 106 L115 110" stroke-width="2.5"/>
    <path d="M159 61 L171 84 M148 72 L161 96 M137 86 L150 111 M126 99 L138 124" stroke-width="2.4"/>

    <path d="M181 151 C155 160 133 174 115 192 L102 205 L88 201 L89 216 L71 216 L76 230 L57 235 L68 246 L49 256 L65 264 L49 279 L72 277 L67 292 L91 283 L94 297 L116 282 L124 293 L143 271 C167 239 181 201 181 151 Z" stroke-width="5"/>
    <path d="M181 151 C154 180 132 208 111 237 C98 255 87 270 73 284" stroke-width="3"/>
    <path d="M165 176 L140 173 M154 191 L128 189 M143 207 L116 206 M132 223 L106 224 M121 240 L95 244 M109 256 L86 262" stroke-width="2.5"/>
    <path d="M151 182 L163 204 M139 198 L151 220 M126 215 L138 237 M114 232 L125 252 M101 249 L111 267" stroke-width="2.4"/>

    <path d="M188 151 C217 142 244 141 267 150 L282 157 L291 148 L298 163 L313 159 L311 175 L327 179 L318 191 L332 201 L319 211 L328 226 L310 229 L312 245 L293 242 L290 258 L271 249 L263 263 L246 247 L235 258 L222 235 C206 208 195 181 188 151 Z" stroke-width="5"/>
    <path d="M188 151 C216 168 238 187 258 208 C274 225 287 241 300 254" stroke-width="3"/>
    <path d="M208 161 L230 153 M219 172 L244 164 M231 183 L257 175 M243 195 L270 187 M255 207 L281 200 M267 220 L291 215" stroke-width="2.5"/>
    <path d="M218 166 L207 190 M231 177 L220 201 M244 189 L233 213 M257 202 L246 225 M270 214 L259 237" stroke-width="2.4"/>

    <path d="M188 151 C214 133 239 111 261 88 C278 69 293 51 308 34" stroke-width="7"/>
  </g>
</svg>
"""

PERILLA_LOGO_SVG = r"""
<svg viewBox="0 0 360 300" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g stroke="#2a0f2e" stroke-linecap="round" stroke-linejoin="round">
    <path d="M188 151 C167 130 147 112 126 99 L111 90 L119 82 L100 78 L109 69 L88 62 L102 53 L84 44 L105 41 L96 28 L119 31 L116 17 L140 27 L145 11 L164 29 L173 18 L181 41 C190 72 196 110 188 151 Z" fill="#a251ae" stroke-width="4"/>
    <path d="M181 151 C155 160 133 174 115 192 L102 205 L88 201 L89 216 L71 216 L76 230 L57 235 L68 246 L49 256 L65 264 L49 279 L72 277 L67 292 L91 283 L94 297 L116 282 L124 293 L143 271 C167 239 181 201 181 151 Z" fill="#8f439c" stroke-width="4"/>
    <path d="M188 151 C217 142 244 141 267 150 L282 157 L291 148 L298 163 L313 159 L311 175 L327 179 L318 191 L332 201 L319 211 L328 226 L310 229 L312 245 L293 242 L290 258 L271 249 L263 263 L246 247 L235 258 L222 235 C206 208 195 181 188 151 Z" fill="#aa54b5" stroke-width="4"/>
    <g fill="none" stroke="#421943">
      <path d="M181 41 C164 55 151 70 139 88 C128 104 117 120 106 139" stroke-width="3"/>
      <path d="M181 151 C154 180 132 208 111 237 C98 255 87 270 73 284" stroke-width="3"/>
      <path d="M188 151 C216 168 238 187 258 208 C274 225 287 241 300 254" stroke-width="3"/>
      <path d="M159 61 L171 84 M148 72 L161 96 M137 86 L150 111 M126 99 L138 124" stroke-width="2"/>
      <path d="M151 182 L163 204 M139 198 L151 220 M126 215 L138 237 M114 232 L125 252" stroke-width="2"/>
      <path d="M218 166 L207 190 M231 177 L220 201 M244 189 L233 213 M257 202 L246 225" stroke-width="2"/>
    </g>
    <path d="M188 151 C214 133 239 111 261 88 C278 69 293 51 308 34" fill="none" stroke="#4b1d43" stroke-width="7"/>
  </g>
</svg>
"""

st.set_page_config(page_title="FreshTag", page_icon="🟣", layout="centered", initial_sidebar_state="collapsed")

st.markdown(_html(r"""
<style>
:root{
  --ink:#f7f3f8; --muted:#b0a3b4; --plum:#8f47a0;
}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
body{background:#08060b;}
.stApp{
  color:var(--ink);
  background:
    radial-gradient(circle at 16% 5%,rgba(137,66,160,.16),transparent 29%),
    radial-gradient(circle at 85% 12%,rgba(165,54,124,.11),transparent 25%),
    radial-gradient(circle at 70% 90%,rgba(88,39,109,.10),transparent 31%),
    linear-gradient(145deg,#08060b 0%,#100913 48%,#09070c 100%);
  min-height:100vh; transition:background .5s ease;
}
#MainMenu,header,footer{visibility:hidden;}
.block-container{max-width:760px;padding-top:1rem;padding-bottom:2.35rem;position:relative;z-index:2;}

.botanical-layer{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;}
.perilla-bg{position:absolute;width:420px;color:#ae6bb8;opacity:.055;filter:drop-shadow(0 0 30px rgba(169,78,179,.10));transition:.5s ease;}
.perilla-bg.left{left:-130px;top:115px;transform:rotate(-10deg);}
.perilla-bg.right{right:-145px;bottom:-5px;transform:rotate(168deg) scale(.94);}

.hero{
  position:relative;overflow:hidden;border-radius:30px;padding:29px 31px 28px;margin-bottom:18px;
  background:
    radial-gradient(circle at 82% 5%,rgba(255,255,255,.08),transparent 20%),
    radial-gradient(circle at 20% 100%,rgba(194,70,145,.12),transparent 28%),
    linear-gradient(135deg,rgba(78,31,99,.82),rgba(45,18,57,.92) 50%,rgba(62,19,56,.87));
  border:1px solid rgba(255,255,255,.10);
  box-shadow:0 28px 80px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.075);
  backdrop-filter:blur(18px);
}
.hero:after{content:"";position:absolute;width:280px;height:280px;right:-110px;top:-145px;border-radius:50%;background:radial-gradient(circle,rgba(202,98,192,.17),transparent 66%);}
.brand-row{position:relative;z-index:1;display:flex;align-items:center;gap:14px;}
.brand-mark{
  width:70px;height:70px;display:grid;place-items:center;border-radius:20px;padding:7px;
  background:linear-gradient(145deg,rgba(175,85,183,.28),rgba(100,41,119,.18));
  border:1px solid rgba(255,255,255,.13);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 14px 34px rgba(0,0,0,.23);
}
.brand-mark svg{width:100%;height:100%;display:block;filter:drop-shadow(0 4px 8px rgba(0,0,0,.18));}
.brand-name{font-size:1.88rem;font-weight:920;letter-spacing:-.055em;line-height:1;}
.brand-mini{margin-top:6px;font-size:.70rem;font-weight:750;letter-spacing:.18em;color:rgba(255,255,255,.53);text-transform:uppercase;}
.hero-copy{position:relative;z-index:1;margin-top:28px;max-width:550px;}
.hero-kicker{display:inline-flex;padding:6px 10px;border-radius:999px;margin-bottom:12px;font-size:.66rem;font-weight:850;letter-spacing:.12em;color:#dfcde5;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);text-transform:uppercase;}
.hero-title{margin:0;max-width:550px;font-size:2.18rem;line-height:1.04;font-weight:930;letter-spacing:-.058em;}
.hero-sub{max-width:470px;margin-top:12px;font-size:.93rem;line-height:1.57;color:rgba(255,255,255,.63);}

.action-head{text-align:center;margin:18px 0 12px;}
.action-title{font-size:1.12rem;font-weight:850;letter-spacing:-.025em;}
.action-sub{margin-top:4px;color:#9d90a1;font-size:.84rem;}

div[role="radiogroup"]{
  width:fit-content;margin:0 auto 13px;padding:6px 8px;border-radius:17px;
  background:rgba(255,255,255,.09)!important;border:1px solid rgba(255,255,255,.14);
  box-shadow:0 14px 36px rgba(0,0,0,.18);backdrop-filter:blur(16px);
}
div[role="radiogroup"] label,
div[role="radiogroup"] label p,
div[role="radiogroup"] [data-testid="stMarkdownContainer"]{
  color:#fff!important;opacity:1!important;font-weight:750!important;
}

div[data-testid="stCameraInput"],div[data-testid="stFileUploader"]{
  overflow:hidden;padding:8px;border-radius:26px;background:rgba(255,255,255,.065);
  border:1px solid rgba(255,255,255,.11);
  box-shadow:0 28px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.045);
  backdrop-filter:blur(18px);
}
div[data-testid="stCameraInput"] video{max-height:420px!important;object-fit:cover!important;border-radius:20px!important;background:#030204!important;}

div[data-testid="stCameraInput"] button{
  min-height:48px!important;border-radius:15px!important;
  border:1px solid rgba(235,159,242,.55)!important;
  background:linear-gradient(135deg,#934aaf,#bd4f96)!important;
  color:#fff!important;font-weight:900!important;opacity:1!important;
  box-shadow:0 9px 28px rgba(148,53,164,.26)!important;
}
div[data-testid="stCameraInput"] button:hover{
  filter:brightness(1.12)!important;transform:translateY(-1px);
}
div[data-testid="stCameraInput"] button *,
div[data-testid="stCameraInput"] button svg{color:#fff!important;fill:#fff!important;opacity:1!important;}

div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"]{
  background:linear-gradient(135deg,#6c1d31,#8f273e)!important;
  border:1px solid rgba(255,107,128,.68)!important;
  color:#ffe4e8!important;
}
div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"] *{color:#ffe4e8!important;fill:#ffe4e8!important;}

div[data-testid="stFileUploader"] button{
  border-radius:14px!important;border:1px solid rgba(226,145,235,.48)!important;
  background:linear-gradient(135deg,#80409a,#a4478d)!important;
  color:#fff!important;font-weight:850!important;opacity:1!important;
}
div[data-testid="stFileUploader"] button *{color:#fff!important;fill:#fff!important;}

.result-shell{position:relative;margin-top:18px;padding:1px;border-radius:29px;overflow:hidden;animation:resultIn .36s cubic-bezier(.2,.8,.2,1);}
@keyframes resultIn{from{opacity:0;transform:translateY(8px) scale(.992)}to{opacity:1;transform:translateY(0) scale(1)}}
.result-card{position:relative;z-index:1;border-radius:28px;padding:31px 25px 29px;text-align:center;backdrop-filter:blur(20px);box-shadow:0 26px 75px rgba(0,0,0,.30);}
.status-mark{width:80px;height:80px;margin:0 auto 16px;display:grid;place-items:center;border-radius:25px;font-size:39px;font-weight:930;line-height:1;}
.status-caption{margin-bottom:7px;font-size:.67rem;font-weight:850;letter-spacing:.18em;text-transform:uppercase;opacity:.58;}
.status-title{font-size:1.64rem;line-height:1.12;font-weight:930;letter-spacing:-.04em;}
.status-message{max-width:430px;margin:10px auto 0;font-size:.92rem;line-height:1.58;opacity:.75;}

.result-shell.fresh{background:linear-gradient(135deg,rgba(60,218,135,.76),rgba(76,126,105,.22));box-shadow:0 0 60px rgba(44,199,118,.16);}
.fresh .result-card{background:radial-gradient(circle at 50% -10%,rgba(43,180,106,.14),transparent 34%),linear-gradient(155deg,rgba(8,30,22,.98),rgba(8,17,16,.97));border:1px solid rgba(91,229,157,.17);}
.fresh .status-mark{color:#72edb0;background:rgba(62,214,139,.12);border:1px solid rgba(103,233,169,.23);box-shadow:0 0 40px rgba(50,205,130,.15);}
.fresh .status-title{color:#82efba;}

.result-shell.warning{background:linear-gradient(135deg,rgba(255,179,63,.88),rgba(142,83,27,.36));box-shadow:0 0 66px rgba(255,159,40,.20);}
.warning .result-card{background:radial-gradient(circle at 50% -10%,rgba(225,143,35,.17),transparent 34%),linear-gradient(155deg,rgba(42,28,9,.98),rgba(20,15,9,.97));border:1px solid rgba(255,196,91,.19);}
.warning .status-mark{color:#ffc45f;background:rgba(255,175,54,.13);border:1px solid rgba(255,195,92,.26);box-shadow:0 0 42px rgba(255,165,43,.18);}
.warning .status-title{color:#ffd178;}

.result-shell.danger{background:linear-gradient(135deg,rgba(255,64,82,.95),rgba(120,18,40,.62));box-shadow:0 0 85px rgba(255,36,65,.30),0 28px 85px rgba(0,0,0,.40);}
.danger .result-card{background:radial-gradient(circle at 50% -12%,rgba(218,29,61,.24),transparent 37%),linear-gradient(155deg,rgba(48,7,15,.99),rgba(17,5,9,.985));border:1px solid rgba(255,87,104,.25);}
.danger .status-mark{color:#ff6a78;background:rgba(255,55,78,.14);border:1px solid rgba(255,100,118,.32);box-shadow:0 0 52px rgba(255,42,67,.28),inset 0 0 26px rgba(255,67,86,.06);}
.danger .status-title{color:#ff7a87;text-shadow:0 0 26px rgba(255,55,76,.18);}

.note{max-width:540px;margin:15px auto 0;text-align:center;color:#756a79;font-size:.68rem;line-height:1.45;}

@media(max-width:620px){
  .block-container{padding-top:.62rem;padding-left:.85rem;padding-right:.85rem;}
  .hero{padding:22px 20px 22px;border-radius:25px;}
  .brand-mark{width:62px;height:62px;border-radius:18px;}
  .brand-name{font-size:1.62rem}.hero-title{font-size:1.73rem}.hero-sub{font-size:.88rem}
  .perilla-bg{width:290px;opacity:.043}.perilla-bg.left{left:-110px;top:180px}.perilla-bg.right{right:-115px;bottom:55px}
  .result-card{padding:27px 17px 24px}.status-title{font-size:1.4rem}
  div[role="radiogroup"]{width:100%}
}
</style>
"""), unsafe_allow_html=True)

st.markdown(_html(f"""
<div class="botanical-layer" aria-hidden="true">
  <div class="perilla-bg left">{PERILLA_CLUSTER_SVG}</div>
  <div class="perilla-bg right">{PERILLA_CLUSTER_SVG}</div>
</div>
"""), unsafe_allow_html=True)

def crop_roi(image, roi):
    w, h = image.size
    x1, y1, x2, y2 = roi
    return image.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))

def median_rgb(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(round(v)) for v in rgb)

def make_features(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return np.array([[r, g, b, h*360, s*100, v*100]], dtype=float)

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
    return {
        "fresh":"fresh","còn tươi":"fresh","con tuoi":"fresh","0":"fresh",
        "transition":"transition","chuyển tiếp":"transition","chuyen tiep":"transition",
        "nên sử dụng sớm":"transition","nen su dung som":"transition","1":"transition",
        "spoiled":"spoiled","hư hỏng":"spoiled","hu hong":"spoiled",
        "có dấu hiệu hư hỏng":"spoiled","co dau hieu hu hong":"spoiled","2":"spoiled",
    }.get(t)

def demo_predict(rgb):
    arr = np.asarray(rgb, dtype=float)
    return min(DEMO_CENTROIDS, key=lambda k: np.linalg.norm(arr - DEMO_CENTROIDS[k]))

def analyze(image):
    rgb = median_rgb(crop_roi(image, INDICATOR_ROI))
    model = load_model()
    if model is None:
        return demo_predict(rgb)
    label = normalize_label(model.predict(make_features(rgb))[0])
    if label is None:
        raise ValueError("Nhãn đầu ra của mô hình không hợp lệ.")
    return label

def apply_result_environment(label):
    if label == "fresh":
        bg = "radial-gradient(circle at 50% 18%,rgba(28,183,105,.18),transparent 31%),radial-gradient(circle at 12% 5%,rgba(96,60,145,.15),transparent 27%),linear-gradient(145deg,#050b08 0%,#081510 48%,#07080b 100%)"
        leaf, opacity = "#78cda1", ".058"
    elif label == "transition":
        bg = "radial-gradient(circle at 50% 18%,rgba(224,141,31,.21),transparent 31%),radial-gradient(circle at 12% 5%,rgba(102,61,143,.14),transparent 27%),linear-gradient(145deg,#0b0804 0%,#181006 48%,#0b0809 100%)"
        leaf, opacity = "#d4a95b", ".061"
    else:
        bg = "radial-gradient(circle at 50% 18%,rgba(210,22,53,.29),transparent 32%),radial-gradient(circle at 12% 5%,rgba(123,22,51,.25),transparent 29%),radial-gradient(circle at 88% 82%,rgba(180,18,46,.19),transparent 32%),linear-gradient(145deg,#080405 0%,#1c070d 48%,#080507 100%)"
        leaf, opacity = "#d45368", ".072"
    st.markdown(_html(f"<style>.stApp{{background:{bg};}}.perilla-bg{{color:{leaf};opacity:{opacity};}}</style>"), unsafe_allow_html=True)

def show_result(label):
    r = RESULTS[label]
    st.markdown(_html(f"""
    <div class="result-shell {r['class']}">
      <div class="result-card">
        <div class="status-mark">{r['symbol']}</div>
        <div class="status-caption">Trạng thái</div>
        <div class="status-title">{r['title']}</div>
        <div class="status-message">{r['message']}</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

st.markdown(_html(f"""
<div class="hero">
  <div class="brand-row">
    <div class="brand-mark">{PERILLA_LOGO_SVG}</div>
    <div>
      <div class="brand-name">FreshTag</div>
      <div class="brand-mini">Freshness indicator</div>
    </div>
  </div>
  <div class="hero-copy">
    <div class="hero-kicker">THẺ CHỈ THỊ TÍA TÔ</div>
    <div class="hero-title">Chụp một lần.<br>Biết trạng thái ngay.</div>
    <div class="hero-sub">Kiểm tra nhanh trạng thái thực phẩm bằng màu của thẻ chỉ thị.</div>
  </div>
</div>
<div class="action-head">
  <div class="action-title">Kiểm tra ngay</div>
  <div class="action-sub">Chụp thẻ chỉ thị hoặc chọn ảnh có sẵn</div>
</div>
"""), unsafe_allow_html=True)

mode = st.radio("Nguồn ảnh", ["📷 Chụp ảnh", "▣ Thư viện"], horizontal=True, label_visibility="collapsed")

if mode == "📷 Chụp ảnh":
    image_file = st.camera_input("Chụp ảnh", label_visibility="collapsed")
else:
    image_file = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"], label_visibility="collapsed")

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        label = analyze(image)
        apply_result_environment(label)
        show_result(label)
    except Exception:
        st.error("Không thể đọc ảnh. Vui lòng chụp lại hoặc chọn ảnh khác.")

st.markdown(_html("""
<div class="note">
Kết quả mang tính hỗ trợ nhận định và không thay thế kiểm nghiệm an toàn thực phẩm.
</div>
"""), unsafe_allow_html=True)
