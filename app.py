import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FetusVision AI — Ritesh Trivedi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import urllib.request
        from ultralytics import YOLO

        # Hugging Face se direct best.pt download
        HF_URL = "https://huggingface.co/ritesht04/Fetus_Detection/resolve/a6282a5fe675300a50c53629738a9583888634c9/best.pt"

        # Agar best.pt nahi hai to download karo
        if not os.path.exists("best.pt"):
            with st.spinner("Model download ho raha hai... pehli baar thoda time lagega"):
                urllib.request.urlretrieve(HF_URL, "best.pt")

        # PIL only mode - cv2/libGL nahi chahiye
        import os
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        model = YOLO("best.pt")
        return model, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, False

model, model_loaded = load_model()

# ─── CSS STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:    #040C18;
    --bg-card:    #071428;
    --bg-glass:   rgba(10, 25, 55, 0.85);
    --accent:     #00D4FF;
    --accent2:    #0066FF;
    --accent3:    #7B2FFF;
    --gold:       #F5C842;
    --success:    #00E5A0;
    --danger:     #FF4567;
    --text-main:  #E8F4FD;
    --text-muted: #6B8BAA;
    --border:     rgba(0, 212, 255, 0.15);
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stApp {
    background: 
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(0,102,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(123,47,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(0,212,255,0.04) 0%, transparent 70%),
        var(--bg-deep) !important;
    min-height: 100vh;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent2); border-radius: 3px; }

/* ── Header / Hero ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,102,255,0.12));
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.35rem 1.2rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.4rem, 5vw, 4rem) !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--accent) 50%, var(--accent2) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    padding: 0 !important;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin-top: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-author {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 1.4rem;
    padding: 0.5rem 1.2rem;
    background: rgba(245,200,66,0.07);
    border: 1px solid rgba(245,200,66,0.2);
    border-radius: 50px;
    font-size: 0.85rem;
    color: var(--gold);
    font-weight: 500;
}
.divider-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), var(--accent), var(--border), transparent);
    margin: 2rem 0;
}

/* ── Metric Cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0 2rem;
}
.metric-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(0,212,255,0.35); }
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.85rem;
    font-weight: 500;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--success);
    margin-top: 0.2rem;
    font-weight: 500;
}

/* ── Upload Zone ── */
.upload-section {
    background: var(--bg-glass);
    border: 2px dashed rgba(0,212,255,0.25);
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-section:hover { border-color: var(--accent); }
.upload-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 0.4rem;
}
.upload-hint { font-size: 0.82rem; color: var(--text-muted); }

/* ── Result Card ── */
.result-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    margin-top: 1rem;
}
.result-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
}
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.3rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.badge-detected {
    background: rgba(0,229,160,0.12);
    border: 1px solid rgba(0,229,160,0.35);
    color: var(--success);
}
.badge-not-detected {
    background: rgba(255,69,103,0.10);
    border: 1px solid rgba(255,69,103,0.30);
    color: var(--danger);
}
.conf-bar-wrap {
    margin: 1rem 0;
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.8s ease;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.stat-key { color: var(--text-muted); }
.stat-val { color: var(--text-main); font-family: 'DM Mono', monospace; font-weight: 500; }

/* ── Info Panel ── */
.info-panel {
    background: rgba(0,102,255,0.06);
    border: 1px solid rgba(0,102,255,0.18);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.info-panel-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent2);
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* ── Tips Panel ── */
.tip-item {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.83rem;
    color: var(--text-muted);
}
.tip-icon { color: var(--accent); flex-shrink: 0; margin-top: 1px; }

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    padding: 2.5rem 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    color: var(--text-muted);
    font-size: 0.8rem;
}
.footer-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--gold);
    font-weight: 700;
    margin-bottom: 0.3rem;
}

/* ── Streamlit Widget Overrides ── */
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
}
.stFileUploader label { color: var(--text-muted) !important; font-size: 0.82rem !important; }
section[data-testid="stFileUploadDropzone"] {
    background: rgba(0,212,255,0.04) !important;
    border: 1.5px dashed rgba(0,212,255,0.25) !important;
    border-radius: 14px !important;
    color: var(--text-muted) !important;
}
.stSlider > div > div > div > div { background: var(--accent2) !important; }
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
div[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent2), var(--accent)) !important;
    border-radius: 50px !important;
}
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
}
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important; }
.stMarkdown p { color: var(--text-muted) !important; font-size: 0.9rem !important; }
.stSidebar { background: var(--bg-card) !important; }
</style>
""", unsafe_allow_html=True)


# ─── HERO SECTION ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🩺 &nbsp; AI-Powered Medical Imaging</div>
    <h1 class="hero-title">FetusVision AI</h1>
    <p class="hero-sub">Ultrasound Image Mein Fetus Detection — YOLOv8 Deep Learning Model</p>
    <div class="hero-author">
        <span>⭐</span>
        <span>Developed by <strong>Ritesh Trivedi</strong></span>
    </div>
</div>
<div class="divider-line"></div>
""", unsafe_allow_html=True)

# ─── METRICS ROW ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-val">99.06%</div>
        <div class="metric-label">mAP @ 50</div>
        <div class="metric-sub">↑ Excellent</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">97.76%</div>
        <div class="metric-label">Precision</div>
        <div class="metric-sub">↑ Outstanding</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">94.64%</div>
        <div class="metric-label">Recall</div>
        <div class="metric-sub">↑ High</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">73/80</div>
        <div class="metric-label">Epochs</div>
        <div class="metric-sub">Early Stop</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    # Upload Section
    st.markdown("""
    <div class="upload-section">
        <div class="upload-icon">🔬</div>
        <div class="upload-title">Ultrasound Image Upload Karo</div>
        <div class="upload-hint">JPG, JPEG, PNG — 2D B-mode Ultrasound Supported</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Image select karo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Settings
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">⚙️ &nbsp; Detection Settings</div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Kitna confident hona chahiye model — zyada matlab sirf high confidence detections"
    )
    iou_threshold = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Duplicate box removal threshold"
    )

    # Tips
    st.markdown("""
    <div class="result-card" style="margin-top:1.5rem;">
        <div class="info-panel-title" style="color:var(--accent);letter-spacing:2px;font-size:0.72rem;font-weight:600;text-transform:uppercase;margin-bottom:0.8rem;">
            💡 &nbsp; Best Results Ke Liye Tips
        </div>
        <div class="tip-item"><span class="tip-icon">›</span><span>2D B-mode standard ultrasound image use karo</span></div>
        <div class="tip-item"><span class="tip-icon">›</span><span>Clear, well-lit scan image better results deta hai</span></div>
        <div class="tip-item"><span class="tip-icon">›</span><span>11+ weeks pregnancy pe best accuracy milti hai</span></div>
        <div class="tip-item"><span class="tip-icon">›</span><span>Confidence 0.25 default — early pregnancy mein 0.15 try karo</span></div>
        <div class="tip-item" style="border:none;"><span class="tip-icon">›</span><span>Model sirf detection karta hai — diagnosis doctor karega</span></div>
    </div>
    """, unsafe_allow_html=True)


with col_right:
    if uploaded_file is None:
        # Placeholder
        st.markdown("""
        <div class="result-card" style="min-height:420px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; gap:1rem;">
            <div style="font-size:4rem; opacity:0.3;">🩺</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:var(--text-muted);">
                Image Upload Karo
            </div>
            <div style="font-size:0.82rem; color:var(--text-muted); max-width:240px; line-height:1.6;">
                Ultrasound image upload karne ke baad AI automatically fetus detect karega
            </div>
            <div style="margin-top:0.5rem; padding:0.5rem 1.2rem; border-radius:50px; border:1px solid var(--border); font-size:0.75rem; color:var(--accent); letter-spacing:1px;">
                YOLOv8 Medium Model Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Show uploaded image
        image = Image.open(uploaded_file)
        img_w, img_h = image.size

        if not model_loaded:
            st.markdown("""
            <div class="result-card">
                <div style="color:var(--danger); font-size:1rem; font-weight:600; margin-bottom:0.5rem;">
                    ⚠️ Model Load Error
                </div>
                <div style="color:var(--text-muted); font-size:0.85rem; line-height:1.6;">
                    best.pt file nahi mili. Please ensure ki best.pt same folder mein hai jahan app.py hai.
                    <br><br>
                    <code style="color:var(--accent); font-family:'DM Mono',monospace;">app.py ke saath best.pt rakho</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            # Run detection
            with st.spinner("🔍 Analyzing ultrasound..."):
                progress_bar = st.progress(0)
                for i in range(0, 101, 20):
                    time.sleep(0.04)
                    progress_bar.progress(i)

                # Convert to RGB if needed
                img_rgb = image.convert("RGB")
                img_array = np.array(img_rgb)

                results = model.predict(
                    source=img_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                progress_bar.progress(100)
                time.sleep(0.1)
                progress_bar.empty()

            # Get result
            result = results[0]
            boxes = result.boxes
            # PIL se annotate karo — cv2/libGL ki zaroorat nahi
            annotated = result.plot(img=img_array)
            annotated_pil = Image.fromarray(annotated[:, :, ::-1])

            # Show annotated image
            st.image(annotated_pil, caption="🔬 Detection Result", use_container_width=True)

            # Detection info
            num_detected = len(boxes) if boxes is not None else 0

            if num_detected > 0:
                best_conf = float(boxes.conf.max()) if len(boxes) > 0 else 0
                conf_pct = best_conf * 100

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <span style="font-size:1.3rem;">✅</span>
                        <span class="result-title">Fetus Detected!</span>
                    </div>
                    <div class="detection-badge badge-detected">
                        🟢 &nbsp; FETUS DETECTED — {num_detected} Instance(s) Found
                    </div>

                    <div style="margin: 1rem 0 0.4rem; font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px;">
                        Best Confidence Score
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size:2rem; color:var(--accent); font-weight:500;">
                        {conf_pct:.1f}%
                    </div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
                    </div>

                    <div class="stat-row">
                        <span class="stat-key">Detections Found</span>
                        <span class="stat-val">{num_detected}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">Confidence Threshold</span>
                        <span class="stat-val">{conf_threshold:.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">IoU Threshold</span>
                        <span class="stat-val">{iou_threshold:.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">Image Size</span>
                        <span class="stat-val">{img_w} × {img_h} px</span>
                    </div>
                    <div class="stat-row" style="border:none;">
                        <span class="stat-key">Model</span>
                        <span class="stat-val">YOLOv8m — best.pt</span>
                    </div>

                    <div style="margin-top:1rem; padding:0.8rem; background:rgba(0,229,160,0.06); border:1px solid rgba(0,229,160,0.15); border-radius:10px; font-size:0.8rem; color:#5EEAD4; line-height:1.6;">
                        ⚕️ &nbsp;<strong>Medical Disclaimer:</strong> Yeh AI tool sirf detection assistance ke liye hai. Final clinical diagnosis hamesha qualified doctor karega.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Per-box details if multiple
                if num_detected > 1:
                    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                    for i, box in enumerate(boxes):
                        c = float(box.conf)
                        st.markdown(f"""
                        <div class="stat-row">
                            <span class="stat-key">Detection #{i+1}</span>
                            <span class="stat-val" style="color:var(--accent);">Conf: {c:.3f} ({c*100:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <span style="font-size:1.3rem;">❌</span>
                        <span class="result-title">No Detection</span>
                    </div>
                    <div class="detection-badge badge-not-detected">
                        🔴 &nbsp; Fetus Detect Nahi Hua
                    </div>

                    <div style="margin-top:1rem; font-size:0.85rem; color:var(--text-muted); line-height:1.7;">
                        Current confidence threshold <strong style="color:var(--text-main);">{conf_threshold}</strong> pe koi fetus detect nahi hua.
                    </div>

                    <div style="margin-top:1rem; padding:0.9rem; background:rgba(255,69,103,0.06); border:1px solid rgba(255,69,103,0.2); border-radius:10px; font-size:0.82rem; color:#FDA4AF; line-height:1.6;">
                        <strong>Possible Reasons:</strong><br>
                        • Early pregnancy (4–7 weeks) — fetus abhi clearly visible nahi<br>
                        • Image quality poor hai<br>
                        • Confidence threshold kam karo (0.15 try karo)<br>
                        • 3D/Doppler image nahi chalegi — 2D B-mode chahiye
                    </div>

                    <div class="stat-row" style="margin-top:1rem;">
                        <span class="stat-key">Image Size</span>
                        <span class="stat-val">{img_w} × {img_h} px</span>
                    </div>
                    <div class="stat-row" style="border:none;">
                        <span class="stat-key">Threshold Used</span>
                        <span class="stat-val">{conf_threshold:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Download annotated image
            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Result Download Karo",
                data=buf.getvalue(),
                file_name="fetus_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


# ─── MODEL INFO SECTION ──────────────────────────────────────────────────────
st.markdown("<div class='divider-line' style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem 0 0.5rem;">
    <div style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:var(--text-main);">
        Model Architecture
    </div>
    <div style="font-size:0.85rem; color:var(--text-muted); margin-top:0.3rem;">
        YOLOv8 Medium — Technical Details
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("""
    <div class="result-card">
        <div class="info-panel-title">🧠 &nbsp; Architecture</div>
        <div class="stat-row"><span class="stat-key">Model</span><span class="stat-val">YOLOv8m</span></div>
        <div class="stat-row"><span class="stat-key">Backbone</span><span class="stat-val">CSPDarknet</span></div>
        <div class="stat-row"><span class="stat-key">Neck</span><span class="stat-val">FPN + PAN</span></div>
        <div class="stat-row"><span class="stat-key">Head</span><span class="stat-val">Decoupled</span></div>
        <div class="stat-row"><span class="stat-key">Parameters</span><span class="stat-val">25.9M</span></div>
        <div class="stat-row" style="border:none;"><span class="stat-key">Input Size</span><span class="stat-val">640 × 640</span></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="result-card">
        <div class="info-panel-title">📊 &nbsp; Training Config</div>
        <div class="stat-row"><span class="stat-key">Epochs</span><span class="stat-val">73 / 80</span></div>
        <div class="stat-row"><span class="stat-key">Optimizer</span><span class="stat-val">AdamW</span></div>
        <div class="stat-row"><span class="stat-key">LR Schedule</span><span class="stat-val">Cosine</span></div>
        <div class="stat-row"><span class="stat-key">Activation</span><span class="stat-val">SiLU</span></div>
        <div class="stat-row"><span class="stat-key">Batch Size</span><span class="stat-val">16</span></div>
        <div class="stat-row" style="border:none;"><span class="stat-key">GPU</span><span class="stat-val">NVIDIA T4</span></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="result-card">
        <div class="info-panel-title">📈 &nbsp; Dataset</div>
        <div class="stat-row"><span class="stat-key">Source</span><span class="stat-val">Roboflow</span></div>
        <div class="stat-row"><span class="stat-key">Train</span><span class="stat-val">2,374 images</span></div>
        <div class="stat-row"><span class="stat-key">Validation</span><span class="stat-val">738 images</span></div>
        <div class="stat-row"><span class="stat-key">Test</span><span class="stat-val">136 images</span></div>
        <div class="stat-row"><span class="stat-key">Classes</span><span class="stat-val">1 — Fetus</span></div>
        <div class="stat-row" style="border:none;"><span class="stat-key">Format</span><span class="stat-val">YOLOv8</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
    <div class="footer-name">Ritesh Trivedi</div>
    <div style="color:var(--text-muted); margin-bottom:0.4rem;">
        Fetus Detection using YOLOv8 Deep Learning
    </div>
    <div style="color:var(--text-muted); font-size:0.75rem; margin-top:0.5rem;">
        ⚕️ For research & educational purposes only &nbsp;|&nbsp; Not a clinical diagnostic tool
    </div>
    <div style="margin-top:1rem; display:flex; justify-content:center; gap:1.5rem; font-size:0.75rem; color:var(--accent);">
        <span>mAP@50: 99.06%</span>
        <span>•</span>
        <span>Precision: 97.76%</span>
        <span>•</span>
        <span>Recall: 94.64%</span>
    </div>
</div>
""", unsafe_allow_html=True)
