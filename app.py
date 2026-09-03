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
    <path d="M180 160 C215 140 246 113 278 78" stroke-width="6"/>

    <path d="M178 154 C154 135 133 117 113 98 L101 88 L111 82 L94 74 L106 67 L88 56 L104 50 L91 38 L111 37 L105 24 L128 29 L129 15 L148 28 L158 18 L168 37 C181 68 188 108 178 154 Z" stroke-width="5"/>
    <path d="M178 154 C166 125 158 96 154 67 C152 51 151 38 148 28" stroke-width="3.2"/>
    <path d="M165 120 L141 104 M161 103 L136 88 M158 86 L134 72 M156 70 L136 55" stroke-width="2.2"/>
    <path d="M166 119 L181 98 M161 101 L177 81 M157 83 L172 65 M154 66 L166 51" stroke-width="2.2"/>

    <path d="M176 159 C151 168 129 182 111 200 L100 211 L86 208 L88 222 L70 223 L76 236 L58 242 L70 252 L53 263 L70 270 L56 285 L78 282 L75 296 L97 286 L102 299 L121 282 L130 292 L146 268 C166 237 178 198 176 159 Z" stroke-width="5"/>
    <path d="M176 159 C153 187 133 213 117 238 C104 258 94 275 78 291" stroke-width="3.2"/>
    <path d="M157 185 L132 183 M147 199 L120 198 M137 214 L109 214 M126 230 L100 233 M116 246 L91 252" stroke-width="2.2"/>
    <path d="M151 187 L164 209 M139 202 L152 224 M127 217 L140 239 M115 233 L127 254 M104 249 L114 268" stroke-width="2.2"/>

    <path d="M184 156 C214 145 242 145 267 154 L281 160 L290 151 L298 166 L313 162 L312 178 L329 181 L320 194 L334 203 L321 214 L331 229 L313 232 L316 248 L297 245 L294 261 L274 251 L266 265 L248 249 L237 260 L223 237 C207 211 195 183 184 156 Z" stroke-width="5"/>
    <path d="M184 156 C213 172 237 190 258 211 C275 228 289 244 304 258" stroke-width="3.2"/>
    <path d="M209 165 L230 156 M221 176 L245 167 M233 187 L259 178 M246 199 L272 190 M258 211 L284 204" stroke-width="2.2"/>
    <path d="M214 168 L204 191 M227 179 L216 202 M240 190 L229 214 M253 202 L242 225 M266 215 L255 237" stroke-width="2.2"/>
  </g>
</svg>
"""

PERILLA_LOGO_SVG = r"""
<svg viewBox="0 0 360 300" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g stroke="#35123b" stroke-linecap="round" stroke-linejoin="round">
    <path d="M180 160 C215 140 246 113 278 78" fill="none" stroke="#4d1e4e" stroke-width="7"/>

    <path d="M178 154 C154 135 133 117 113 98 L101 88 L111 82 L94 74 L106 67 L88 56 L104 50 L91 38 L111 37 L105 24 L128 29 L129 15 L148 28 L158 18 L168 37 C181 68 188 108 178 154 Z" fill="#9d4ca9" stroke-width="4"/>
    <path d="M176 159 C151 168 129 182 111 200 L100 211 L86 208 L88 222 L70 223 L76 236 L58 242 L70 252 L53 263 L70 270 L56 285 L78 282 L75 296 L97 286 L102 299 L121 282 L130 292 L146 268 C166 237 178 198 176 159 Z" fill="#853a92" stroke-width="4"/>
    <path d="M184 156 C214 145 242 145 267 154 L281 160 L290 151 L298 166 L313 162 L312 178 L329 181 L320 194 L334 203 L321 214 L331 229 L313 232 L316 248 L297 245 L294 261 L274 251 L266 265 L248 249 L237 260 L223 237 C207 211 195 183 184 156 Z" fill="#a84fb2" stroke-width="4"/>

    <g fill="none" stroke="#4b1a4f">
      <path d="M178 154 C166 125 158 96 154 67 C152 51 151 38 148 28" stroke-width="3"/>
      <path d="M176 159 C153 187 133 213 117 238 C104 258 94 275 78 291" stroke-width="3"/>
      <path d="M184 156 C213 172 237 190 258 211 C275 228 289 244 304 258" stroke-width="3"/>
      <path d="M165 120 L141 104 M161 103 L136 88 M158 86 L134 72 M156 70 L136 55 M166 119 L181 98 M161 101 L177 81 M157 83 L172 65" stroke-width="2"/>
      <path d="M157 185 L132 183 M147 199 L120 198 M137 214 L109 214 M126 230 L100 233 M151 187 L164 209 M139 202 L152 224 M127 217 L140 239" stroke-width="2"/>
      <path d="M209 165 L230 156 M221 176 L245 167 M233 187 L259 178 M246 199 L272 190 M214 168 L204 191 M227 179 L216 202 M240 190 L229 214" stroke-width="2"/>
    </g>
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
.hero:before{
  content:"";
  position:absolute;
  width:420px;
  height:420px;
  left:-160px;
  bottom:-250px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(187,71,168,.14),rgba(187,71,168,0) 68%);
  filter:blur(8px);
  animation:heroGlow 9s ease-in-out infinite alternate;
}
@keyframes heroGlow{
  from{transform:translate3d(0,0,0) scale(1);opacity:.55;}
  to{transform:translate3d(70px,-28px,0) scale(1.08);opacity:1;}
}

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
.hero-copy{position:relative;z-index:1;margin-top:34px;max-width:590px;}
.hero-kicker{display:none;}
.hero-title{margin:0;max-width:590px;font-size:2.42rem;line-height:.98;font-weight:940;letter-spacing:-.065em;text-wrap:balance;}
.hero-sub{max-width:470px;margin-top:16px;font-size:.92rem;line-height:1.5;color:rgba(255,255,255,.62);letter-spacing:.01em;}

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

div[data-testid="stFileUploader"]{
  overflow:hidden;padding:8px;border-radius:26px;background:rgba(255,255,255,.065);
  border:1px solid rgba(255,255,255,.11);
  box-shadow:0 28px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.045);
  backdrop-filter:blur(18px);
}

div[data-testid="stFileUploader"]{
  max-width:560px;
  margin:0 auto;
}
div[data-testid="stFileUploader"] section{
  min-height:118px!important;
  border:1px dashed rgba(224,154,235,.30)!important;
  border-radius:20px!important;
  background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.018))!important;
}
div[data-testid="stFileUploader"] button{
  min-height:50px!important;
  min-width:170px!important;
  border-radius:16px!important;
  border:1px solid rgba(226,145,235,.55)!important;
  background:linear-gradient(135deg,#85449d,#ad4d92)!important;
  color:#fff!important;
  font-weight:900!important;
  opacity:1!important;
  box-shadow:0 12px 30px rgba(132,57,148,.24)!important;
  transition:filter .18s ease,transform .18s ease,box-shadow .18s ease!important;
}
div[data-testid="stFileUploader"] button:hover{
  filter:brightness(1.12)!important;
  transform:translateY(-1px)!important;
  box-shadow:0 15px 34px rgba(157,67,166,.31)!important;
}
div[data-testid="stFileUploader"] button *,
div[data-testid="stFileUploader"] button svg{
  color:#fff!important;
  fill:#fff!important;
  opacity:1!important;
}
.mobile-capture-note{
  max-width:560px;
  margin:8px auto 0;
  text-align:center;
  color:#978a9b;
  font-size:.76rem;
  line-height:1.45;
}
@media(max-width:640px){
  div[data-testid="stFileUploader"] section{
    min-height:92px!important;
    padding:12px!important;
  }
  div[data-testid="stFileUploader"] section > div{
    gap:8px!important;
  }
  div[data-testid="stFileUploader"] button{
    width:100%!important;
    min-height:52px!important;
  }
}

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
  .brand-name{font-size:1.62rem}.hero-title{font-size:1.9rem}.hero-sub{font-size:.88rem}
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
      <div class="brand-mini">PERILLA COLOR INTELLIGENCE</div>
    </div>
  </div>

  <div class="hero-copy">
    <div class="hero-title">MÀU THAY ĐỔI.<br>TƯƠI HAY THỐI?</div>
    <div class="hero-sub">Chụp thẻ. FreshTag đọc màu.</div>
  </div>
</div>

<div class="action-head">
  <div class="action-title">Quét thẻ</div>
  <div class="action-sub">Một ảnh thẻ là đủ để bắt đầu</div>
</div>
"""), unsafe_allow_html=True)

mode = st.radio(
    "Nguồn ảnh",
    ["📷 Chụp thẻ", "▣ Thư viện"],
    horizontal=True,
    label_visibility="collapsed",
)

# NOTE 11 — MOBILE CAPTURE
# Không dùng st.camera_input() vì trên một số điện thoại luồng webcam/video
# khiến người dùng phải pause rồi chọn frame. Thay vào đó dùng file input ảnh
# của trình duyệt. Trên điện thoại, trình duyệt thường cho phép chọn Camera
# để chụp ảnh tĩnh trực tiếp; trên desktop nó hoạt động như file picker.
#
# Đây là giải pháp không phụ thuộc component ngoài và giữ deploy Streamlit đơn giản.
# Nếu sau này cần ép mở camera sau chỉ bằng 1 lần chạm trên mọi trình duyệt,
# nên chuyển phần capture sang custom component HTML có:
# <input type="file" accept="image/*" capture="environment">

if mode == "📷 Chụp thẻ":
    st.markdown(
        _html("""
        <div class="mobile-capture-note">
          Trên điện thoại, chọn <b>Camera</b> khi bảng chọn ảnh mở ra để chụp thẻ trực tiếp.
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.markdown(
        _html("""
        <style>
        div[data-testid="stFileUploader"] button{
          font-size:0!important;
        }
        div[data-testid="stFileUploader"] button::after{
          content:"MỞ CAMERA";
          font-size:.88rem;
          letter-spacing:.035em;
          color:#fff;
        }
        </style>
        """),
        unsafe_allow_html=True,
    )
    image_file = st.file_uploader(
        "Chụp thẻ",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="capture_photo",
    )
else:
    st.markdown(
        _html("""
        <style>
        div[data-testid="stFileUploader"] button{
          font-size:0!important;
        }
        div[data-testid="stFileUploader"] button::after{
          content:"CHỌN ẢNH";
          font-size:.88rem;
          letter-spacing:.035em;
          color:#fff;
        }
        </style>
        """),
        unsafe_allow_html=True,
    )
    image_file = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="library_photo",
    )

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
