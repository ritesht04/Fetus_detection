import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import time
import os
import urllib.request

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FetusVision AI — Ritesh Trivedi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── LOAD MODEL (Hugging Face se — PIL only, no cv2/libGL) ───────────────────
@st.cache_resource
def load_model():
    try:
        # libGL fix — headless mode
        os.environ["MPLBACKEND"] = "Agg"

        from ultralytics import YOLO

        HF_URL = "https://huggingface.co/ritesht04/Fetus_Detection/resolve/a6282a5fe675300a50c53629738a9583888634c9/best.pt"

        if not os.path.exists("best.pt"):
            urllib.request.urlretrieve(HF_URL, "best.pt")

        model = YOLO("best.pt")
        return model, True
    except Exception as e:
        return None, str(e)

# ─── DRAW BOXES WITH PIL (cv2/libGL nahi chahiye) ────────────────────────────
def draw_boxes_pil(image, boxes, names):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"

        # Box
        draw.rectangle([x1, y1, x2, y2], outline="#00D4FF", width=3)

        # Label background
        tw = len(label) * 7 + 8
        draw.rectangle([x1, y1 - 22, x1 + tw, y1], fill="#00D4FF")

        # Label text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((x1 + 4, y1 - 19), label, fill="#000000", font=font)

    return img

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:    #040C18;
    --bg-card:    #071428;
    --bg-glass:   rgba(10,25,55,0.85);
    --accent:     #00D4FF;
    --accent2:    #0066FF;
    --gold:       #F5C842;
    --success:    #00E5A0;
    --danger:     #FF4567;
    --text-main:  #E8F4FD;
    --text-muted: #6B8BAA;
    --border:     rgba(0,212,255,0.15);
}
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(0,102,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(123,47,255,0.08) 0%, transparent 60%),
        var(--bg-deep) !important;
    min-height: 100vh;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-thumb { background:var(--accent2); border-radius:3px; }

.hero-wrap { text-align:center; padding:3rem 1rem 1.5rem; }
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(0,212,255,0.12),rgba(0,102,255,0.12));
    border:1px solid var(--border); border-radius:50px;
    padding:0.35rem 1.2rem; font-size:0.72rem; font-weight:600;
    letter-spacing:2.5px; text-transform:uppercase; color:var(--accent); margin-bottom:1rem;
}
.hero-title {
    font-family:'Playfair Display',serif !important;
    font-size:clamp(2.2rem,5vw,3.8rem) !important; font-weight:900 !important;
    background:linear-gradient(135deg,#FFFFFF 0%,var(--accent) 50%,var(--accent2) 100%);
    -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important;
    background-clip:text !important; margin:0 !important; line-height:1.1 !important;
}
.hero-sub { font-size:1rem; color:var(--text-muted); margin-top:0.8rem; font-weight:300; }
.hero-author {
    display:inline-flex; align-items:center; gap:0.5rem; margin-top:1.2rem;
    padding:0.45rem 1.2rem; background:rgba(245,200,66,0.07);
    border:1px solid rgba(245,200,66,0.2); border-radius:50px;
    font-size:0.85rem; color:var(--gold); font-weight:500;
}
.divider { height:1px; background:linear-gradient(90deg,transparent,var(--border),var(--accent),var(--border),transparent); margin:1.5rem 0; }

.metrics-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.2rem 0 1.8rem; }
.metric-card {
    background:var(--bg-glass); border:1px solid var(--border); border-radius:16px;
    padding:1.1rem 1rem; text-align:center; backdrop-filter:blur(20px);
    position:relative; overflow:hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--accent2),var(--accent)); }
.metric-val { font-family:'DM Mono',monospace; font-size:1.7rem; font-weight:500; color:var(--accent); line-height:1; }
.metric-label { font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; margin-top:0.3rem; }
.metric-sub { font-size:0.75rem; color:var(--success); margin-top:0.2rem; }

.card {
    background:var(--bg-glass); border:1px solid var(--border);
    border-radius:20px; padding:1.4rem; backdrop-filter:blur(20px); margin-bottom:1rem;
}
.card-title { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; margin-bottom:1rem; }
.section-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; color:var(--accent); font-weight:600; margin-bottom:0.7rem; }

.stat-row { display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.83rem; }
.stat-row:last-child { border-bottom:none; }
.sk { color:var(--text-muted); }
.sv { color:var(--text-main); font-family:'DM Mono',monospace; font-weight:500; }

.badge-ok { display:inline-flex; align-items:center; gap:0.5rem; padding:0.55rem 1.2rem; border-radius:50px; background:rgba(0,229,160,0.12); border:1px solid rgba(0,229,160,0.35); color:var(--success); font-size:0.88rem; font-weight:600; }
.badge-no { display:inline-flex; align-items:center; gap:0.5rem; padding:0.55rem 1.2rem; border-radius:50px; background:rgba(255,69,103,0.10); border:1px solid rgba(255,69,103,0.30); color:var(--danger); font-size:0.88rem; font-weight:600; }

.conf-wrap { margin:0.8rem 0; background:rgba(255,255,255,0.05); border-radius:50px; height:7px; overflow:hidden; }
.conf-fill { height:100%; border-radius:50px; background:linear-gradient(90deg,var(--accent2),var(--accent)); }

.tip { display:flex; gap:0.6rem; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.81rem; color:var(--text-muted); }
.tip:last-child { border-bottom:none; }
.ti { color:var(--accent); flex-shrink:0; }

.footer { text-align:center; padding:2rem 1rem; border-top:1px solid var(--border); margin-top:2.5rem; }
.footer-name { font-family:'Playfair Display',serif; font-size:1.1rem; color:var(--gold); font-weight:700; margin-bottom:0.3rem; }

/* Streamlit overrides */
section[data-testid="stFileUploadDropzone"] {
    background:rgba(0,212,255,0.04) !important;
    border:1.5px dashed rgba(0,212,255,0.25) !important;
    border-radius:14px !important;
}
.stSlider > div > div > div > div { background:var(--accent2) !important; }
.stButton > button {
    background:linear-gradient(135deg,var(--accent2),var(--accent)) !important;
    border:none !important; border-radius:10px !important; color:white !important;
    font-weight:600 !important; font-family:'DM Sans',sans-serif !important;
}
div[data-testid="stImage"] img { border-radius:14px !important; border:1px solid var(--border) !important; }
.stProgress > div > div > div { background:linear-gradient(90deg,var(--accent2),var(--accent)) !important; border-radius:50px !important; }
h1,h2,h3 { font-family:'Playfair Display',serif !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🩺 &nbsp; AI-Powered Medical Imaging</div>
    <h1 class="hero-title">FetusVision AI</h1>
    <p class="hero-sub">Ultrasound Image Mein Fetus Detection — YOLOv8 Deep Learning</p>
    <div class="hero-author">⭐ &nbsp; Developed by <strong>&nbsp;Ritesh Trivedi</strong></div>
</div>
<div class="divider"></div>
<div class="metrics-row">
    <div class="metric-card"><div class="metric-val">99.06%</div><div class="metric-label">mAP @ 50</div><div class="metric-sub">↑ Excellent</div></div>
    <div class="metric-card"><div class="metric-val">97.76%</div><div class="metric-label">Precision</div><div class="metric-sub">↑ Outstanding</div></div>
    <div class="metric-card"><div class="metric-val">94.64%</div><div class="metric-label">Recall</div><div class="metric-sub">↑ High</div></div>
    <div class="metric-card"><div class="metric-val">73/80</div><div class="metric-label">Epochs</div><div class="metric-sub">Early Stop</div></div>
</div>
""", unsafe_allow_html=True)

# ─── MODEL LOAD (with button) ─────────────────────────────────────────────────
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "model_obj" not in st.session_state:
    st.session_state.model_obj = None

if not st.session_state.model_ready:
    st.markdown("""
    <div class="card" style="text-align:center; padding:2.5rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
        <div class="card-title" style="font-size:1.3rem;">Model Load Karo</div>
        <div style="color:var(--text-muted); font-size:0.88rem; margin-bottom:1.5rem;">
            Hugging Face se YOLOv8 model download hoga — pehli baar 1-2 min lagenge
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        if st.button("⬇️ Model Load Karo — Click Here", use_container_width=True):
            with st.spinner("🔄 Hugging Face se model download ho raha hai..."):
                model, status = load_model()
            if status is True:
                st.session_state.model_ready = True
                st.session_state.model_obj = model
                st.success("✅ Model successfully load ho gaya! Ab image upload karo.")
                st.rerun()
            else:
                st.error(f"❌ Error: {status}")
    st.stop()

# ─── MAIN APP (Model loaded) ──────────────────────────────────────────────────
model = st.session_state.model_obj

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown("""
    <div class="card">
        <div class="section-label">🔬 &nbsp; Image Upload</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700; margin-bottom:0.4rem;">
            Ultrasound Image Select Karo
        </div>
        <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:1rem;">
            JPG, JPEG, PNG — 2D B-mode Ultrasound
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Image upload karo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
        <div class="section-label">⚙️ &nbsp; Detection Settings</div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05,
        help="Zyada = sirf high confidence detections")
    iou_threshold  = st.slider("IoU (NMS) Threshold", 0.20, 0.80, 0.45, 0.05,
        help="Duplicate box removal")

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
        <div class="section-label">💡 &nbsp; Tips</div>
        <div class="tip"><span class="ti">›</span><span>2D B-mode standard ultrasound use karo</span></div>
        <div class="tip"><span class="ti">›</span><span>11+ weeks pregnancy pe best accuracy</span></div>
        <div class="tip"><span class="ti">›</span><span>Early pregnancy mein conf 0.15 try karo</span></div>
        <div class="tip"><span class="ti">›</span><span>Clear scan image better results deta hai</span></div>
        <div class="tip"><span class="ti">›</span><span>Diagnosis doctor se confirm karein</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded_file is None:
        st.markdown("""
        <div class="card" style="min-height:400px; display:flex; flex-direction:column;
             align-items:center; justify-content:center; text-align:center; gap:1rem;">
            <div style="font-size:4rem; opacity:0.25;">🩺</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:var(--text-muted);">
                Image Upload Karo
            </div>
            <div style="font-size:0.82rem; color:var(--text-muted); max-width:220px; line-height:1.6;">
                Left side se image select karo — AI automatically detect karega
            </div>
            <div style="padding:0.4rem 1rem; border-radius:50px; border:1px solid var(--border);
                 font-size:0.72rem; color:var(--success); letter-spacing:1px;">
                ✅ Model Ready
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_w, img_h = image.size

        # Run Detection
        with st.spinner("🔍 Analyzing..."):
            pb = st.progress(0)
            for i in range(0, 80, 20):
                time.sleep(0.03)
                pb.progress(i)

            img_array = np.array(image)
            results = model.predict(
                source=img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            pb.progress(100)
            time.sleep(0.08)
            pb.empty()

        result  = results[0]
        boxes   = result.boxes
        names   = model.names
        num_det = len(boxes) if boxes is not None else 0

        # Draw boxes using PIL — no cv2 needed
        annotated_img = draw_boxes_pil(image, boxes, names)
        st.image(annotated_img, caption="🔬 Detection Result", use_container_width=True)

        # Result Card
        if num_det > 0:
            best_conf = float(boxes.conf.max())
            conf_pct  = best_conf * 100
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;
                     padding-bottom:0.8rem; border-bottom:1px solid var(--border);">
                    <span style="font-size:1.3rem;">✅</span>
                    <span class="card-title" style="margin:0;">Fetus Detected!</span>
                </div>
                <div class="badge-ok">🟢 &nbsp; FETUS DETECTED — {num_det} Instance(s)</div>
                <div style="margin:1rem 0 0.3rem; font-size:0.7rem; color:var(--text-muted);
                     text-transform:uppercase; letter-spacing:1.5px;">Best Confidence</div>
                <div style="font-family:'DM Mono',monospace; font-size:2rem; color:var(--accent); font-weight:500;">
                    {conf_pct:.1f}%
                </div>
                <div class="conf-wrap"><div class="conf-fill" style="width:{conf_pct}%;"></div></div>
                <div class="stat-row"><span class="sk">Detections</span><span class="sv">{num_det}</span></div>
                <div class="stat-row"><span class="sk">Confidence Threshold</span><span class="sv">{conf_threshold:.2f}</span></div>
                <div class="stat-row"><span class="sk">IoU Threshold</span><span class="sv">{iou_threshold:.2f}</span></div>
                <div class="stat-row"><span class="sk">Image Size</span><span class="sv">{img_w} × {img_h} px</span></div>
                <div class="stat-row"><span class="sk">Model</span><span class="sv">YOLOv8m</span></div>
                <div style="margin-top:1rem; padding:0.8rem; background:rgba(0,229,160,0.06);
                     border:1px solid rgba(0,229,160,0.15); border-radius:10px;
                     font-size:0.78rem; color:#5EEAD4; line-height:1.6;">
                    ⚕️ <strong>Disclaimer:</strong> Yeh AI tool sirf assistance ke liye hai.
                    Final diagnosis doctor karega.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if num_det > 1:
                for i, box in enumerate(boxes):
                    c = float(box.conf)
                    st.markdown(f"""
                    <div class="stat-row">
                        <span class="sk">Detection #{i+1}</span>
                        <span class="sv" style="color:var(--accent);">{c*100:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;
                     padding-bottom:0.8rem; border-bottom:1px solid var(--border);">
                    <span style="font-size:1.3rem;">❌</span>
                    <span class="card-title" style="margin:0;">No Detection</span>
                </div>
                <div class="badge-no">🔴 &nbsp; Fetus Detect Nahi Hua</div>
                <div style="margin-top:1rem; padding:0.9rem; background:rgba(255,69,103,0.06);
                     border:1px solid rgba(255,69,103,0.2); border-radius:10px;
                     font-size:0.81rem; color:#FDA4AF; line-height:1.7;">
                    <strong>Possible Reasons:</strong><br>
                    • Early pregnancy (4–7 weeks) — fetus visible nahi<br>
                    • Image quality poor hai<br>
                    • Confidence threshold kam karo (0.15 try karo)<br>
                    • 2D B-mode image chahiye — 3D/Doppler nahi
                </div>
                <div class="stat-row" style="margin-top:1rem;">
                    <span class="sk">Image Size</span><span class="sv">{img_w} × {img_h} px</span>
                </div>
                <div class="stat-row">
                    <span class="sk">Threshold</span><span class="sv">{conf_threshold:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Download
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Result Download Karo",
            data=buf.getvalue(),
            file_name="fetus_result.png",
            mime="image/png",
            use_container_width=True
        )

# ─── MODEL INFO ──────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("""<div class="card">
        <div class="section-label">🧠 &nbsp; Architecture</div>
        <div class="stat-row"><span class="sk">Model</span><span class="sv">YOLOv8m</span></div>
        <div class="stat-row"><span class="sk">Backbone</span><span class="sv">CSPDarknet</span></div>
        <div class="stat-row"><span class="sk">Neck</span><span class="sv">FPN + PAN</span></div>
        <div class="stat-row"><span class="sk">Head</span><span class="sv">Decoupled</span></div>
        <div class="stat-row"><span class="sk">Parameters</span><span class="sv">25.9M</span></div>
        <div class="stat-row"><span class="sk">Input Size</span><span class="sv">640 × 640</span></div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card">
        <div class="section-label">📊 &nbsp; Training</div>
        <div class="stat-row"><span class="sk">Epochs</span><span class="sv">73 / 80</span></div>
        <div class="stat-row"><span class="sk">Optimizer</span><span class="sv">AdamW</span></div>
        <div class="stat-row"><span class="sk">LR Schedule</span><span class="sv">Cosine</span></div>
        <div class="stat-row"><span class="sk">Activation</span><span class="sv">SiLU</span></div>
        <div class="stat-row"><span class="sk">Batch Size</span><span class="sv">16</span></div>
        <div class="stat-row"><span class="sk">GPU</span><span class="sv">NVIDIA T4</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card">
        <div class="section-label">📈 &nbsp; Dataset</div>
        <div class="stat-row"><span class="sk">Source</span><span class="sv">Roboflow</span></div>
        <div class="stat-row"><span class="sk">Train</span><span class="sv">2,374 images</span></div>
        <div class="stat-row"><span class="sk">Validation</span><span class="sv">738 images</span></div>
        <div class="stat-row"><span class="sk">Test</span><span class="sv">136 images</span></div>
        <div class="stat-row"><span class="sk">Classes</span><span class="sv">1 — Fetus</span></div>
        <div class="stat-row"><span class="sk">Format</span><span class="sv">YOLOv8</span></div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-name">Ritesh Trivedi</div>
    <div style="color:var(--text-muted); font-size:0.83rem;">Fetus Detection using YOLOv8 Deep Learning</div>
    <div style="color:var(--text-muted); font-size:0.75rem; margin-top:0.4rem;">
        ⚕️ For research & educational purposes only — Not a clinical diagnostic tool
    </div>
    <div style="margin-top:0.8rem; font-size:0.75rem; color:var(--accent);">
        mAP@50: 99.06% &nbsp;•&nbsp; Precision: 97.76% &nbsp;•&nbsp; Recall: 94.64%
    </div>
</div>
""", unsafe_allow_html=True)
