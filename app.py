import streamlit as st
from PIL import Image
import numpy as np
import os
import requests

st.set_page_config(page_title="FetusVision AI", layout="wide")

# ─── MODEL LOAD ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO

        HF_URL = "https://huggingface.co/ritesht04/Fetus_Detection/resolve/main/best.pt"
        MODEL_PATH = "best.pt"

        # 🔽 Download function
        def download_model(url, path):
            try:
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    return True
                return False
            except Exception as e:
                return False

        # 🔽 Download if not exists
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Model download ho raha hai..."):
                success = download_model(HF_URL, MODEL_PATH)

            if not success:
                st.error("❌ Model download fail ho gaya (Hugging Face se)")
                return None, False

        # 🔽 Load YOLO
        model = YOLO(MODEL_PATH)
        return model, True

    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, False


model, model_loaded = load_model()

# ─── UI ─────────────────────────────────────────────────────
st.title("🩺 FetusVision AI")
st.write("Ultrasound image me fetus detection (YOLOv8)")

uploaded_file = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png"])

conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.25)

# ─── MAIN ───────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if not model_loaded:
        st.error("Model load nahi hua")
    else:
        with st.spinner("🔍 Detect ho raha hai..."):
            img = np.array(image.convert("RGB"))

            results = model.predict(
                source=img,
                conf=conf_threshold,
                verbose=False
            )

        result = results[0]
        boxes = result.boxes

        annotated = result.plot()
        annotated_img = Image.fromarray(annotated[:, :, ::-1])

        st.image(annotated_img, caption="Result", use_container_width=True)

        if boxes is not None and len(boxes) > 0:
            best_conf = float(boxes.conf.max())
            st.success(f"✅ Fetus detected ({best_conf*100:.1f}%)")
        else:
            st.error("❌ Fetus detect nahi hua")

# ─── DEBUG (optional) ───────────────────────────────────────
# st.write(os.listdir())
