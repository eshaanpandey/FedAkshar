import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from app_labels import class_names

# --- Page and Model Setup ---
st.set_page_config(page_title="Federated Devanagari Classifier", layout="wide")

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("federated_devanagari_model.h5")
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

model = load_model()

# --- Inject Custom CSS ---
st.markdown("""
<style>
body, .stApp {
    background-color: #2E2E2E;
    color: #FFFFFF;
}
section.main > div { 
    max-width: 900px; 
    margin: auto; 
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6rem;
}
[data-baseweb="tab-list"] {
    justify-content: center;
    font-size: 1.5rem;
}
.stMetricLabel { font-size:1.5rem !important; }
.stMetricValue { font-size:2rem !important; }
img, video {
    max-width: 600px !important;
    height: auto !important;
    margin: auto;
}
div.stFileUploader, div.stSelectbox {
    width: 600px;
    margin: 30px auto;
    height: 80px;
    font-size: 1.2rem;
}
div.stFileUploader input, div.stSelectbox div[data-baseweb="select"] {
    min-height: 60px;
    font-size: 1.2rem;
}
button {
    font-size: 1.2rem;
}
.stAlert {
    background-color: #444 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- Top Navbar ---
tab_home, tab_about, tab_why = st.tabs(["Home", "About", "Why Federated?"])

# --- HOME PAGE ---
with tab_home:
    st.header("Devanagari Letter Classifier")

    with st.container():
        st.markdown("""
        <div style="background-color: #444; padding: 30px; border-radius: 20px; margin-bottom: 30px;">
            <h4 style="text-align: center; color: white;">Upload a handwritten Devanagari letter.</h4>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose an image", type=["jpg","png"])
        samples = [f for f in os.listdir("samples") if f.lower().endswith((".png",".jpg", ".jpeg"))]
        choice = st.selectbox("Or pick a sample:", [""] + samples)

        st.markdown("</div>", unsafe_allow_html=True)

        if choice and not uploaded:
            uploaded = open(os.path.join("samples", choice), "rb")

        col1, col2 = st.columns(2)
        if uploaded:
            data = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            col1.image(img[:,:,::-1], caption="Input Image", width=500)

            img_rs = cv2.resize(img, (32,32))
            preds = model.predict(np.expand_dims(img_rs, 0))
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            letter = class_names[idx] if 0 <= idx < len(class_names) else "Unknown"

            col2.metric("Predicted Letter", letter)
            col2.metric("Confidence", f"{conf:.2%}")
        else:
            col1.info("Please upload or select an image.")

    st.markdown("### Example Results")

    exs = [f for f in os.listdir("examples") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    cols = st.columns(2)

    for idx, fname in enumerate(exs):
        img_ex = cv2.imread(os.path.join("examples", fname))
        with cols[idx % 2]:
            st.image(img_ex[:, :, ::-1], caption=fname, use_container_width=True)

# --- ABOUT PAGE ---
with tab_about:
    st.header("📋 Project Overview & Technical Details")
    st.subheader("Dataset")
    st.markdown("""
    - **DevanagariHandwrittenCharacterDataset**: 46 classes (36 letters, 10 numerals)  
    - **Images**: 70,400 train / 7,800 test (32×32 px)  
    - **Split**: Data federated across 3 clients  
    """)
    st.subheader("Model Architecture")
    st.markdown("""
    1. **4× Conv2D → BatchNorm → MaxPool**  
    2. **Flatten → Dense(128) → BatchNorm → Dropout**  
    3. **Dense(64) → BatchNorm → Dropout**  
    4. **Dense(46, softmax)**  
    """)
    st.subheader("Key Achievements")
    st.markdown("""
    - **97%** centralized CNN vs **96.2%** federated accuracy  
    - Secure Federated Averaging with privacy  
    - Robust generalization on heterogeneous data  
    """)
    st.subheader("Features & Use Cases")
    st.markdown("""
    - Privacy-preserving OCR for handwritten text  
    - Edge deployment on IoT & mobile devices  
    - Fast <1 s inference, modular codebase  
    """)

# --- WHY FEDERATED? PAGE ---
with tab_why:
    st.header("🔒 Why Federated Learning?")
    st.markdown("""
    1. **Data Privacy:** Raw data never leaves devices  
    2. **Decentralized Training:** No central data bottleneck  
    3. **Faster Training:** ~1/3 the time of single-node ML  
    4. **Heterogeneity Robustness:** Learns diverse handwriting styles  
    5. **Scalable:** Add clients without extra central power  
    """)
    st.image("assets/implementation_diagram.png", caption="Federated Learning Workflow", use_container_width=False)
    # st.video("assets/Federated Learning of Devanagri Script at 3x.mp4", start_time=0)
    # st.markdown("Training the model on 3 nodes")

# --- Footer ---
st.markdown("---")
st.caption("Built with **Streamlit** & **TensorFlow**. © 2025 Your Name")
