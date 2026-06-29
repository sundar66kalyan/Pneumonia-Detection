import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection - AI Powered",
    page_icon="🫁",
    layout="wide"
)

# -----------------------------
# Custom CSS for Animations & Styling
# -----------------------------
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Animated title */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .title-animation {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Floating animation for cards */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-card {
        animation: float 3s ease-in-out infinite;
        border-radius: 20px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    .float-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    /* Pulse animation for prediction */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 1s ease-in-out 3;
    }
    
    /* Glowing button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error cards with animation */
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease;
    }
    
    .error-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress bar animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: progressBar 2s linear infinite;
    }
    
    @keyframes progressBar {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header with Animation
# -----------------------------
st.markdown('<div class="title-animation">🫁 Pneumonia Detection</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 30px;">
    <b>AI-Powered Chest X-Ray Analysis</b> — Detect Pneumonia with 89.10% Accuracy
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar - Project Information
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2>📋 Project Info</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ About This Project", expanded=True):
        st.markdown("""
        **Project:** Pneumonia Chest X-Ray Classification  
        **Code:** PRAICP-1012  
        **Type:** Deep Learning / Computer Vision  
        **Framework:** TensorFlow 2.0  
        **Model:** MobileNetV2 (Fine-Tuned)  
        **Accuracy:** 89.10%  
        **Classes:** NORMAL / PNEUMONIA
        """)
    
    st.markdown("---")
    
    with st.expander("🎯 How It Works", expanded=True):
        st.markdown("""
        1. **Upload** a chest X-ray image (JPG/PNG)  
        2. **Click** the Predict button  
        3. **Get** instant diagnosis with confidence score  
        4. **View** detailed results and analysis  
        
        The model processes the image and predicts with **98%+ confidence** in most cases.
        """)
    
    st.markdown("---")
    
    with st.expander("🖼️ Sample Images", expanded=True):
        st.markdown("""
        **Try these sample images:**
        
        - **NORMAL:** IM-0288-0001.jpeg (439.9KB)  
        - **PNEUMONIA:** person100_virus_166.jpeg  
        
        *Images should be clear chest X-rays (front view).*
        """)
    
    st.markdown("---")
    
    with st.expander("🏥 Clinical Significance", expanded=True):
        st.markdown("""
        - ⚡ **Rapid Screening:** Results in seconds  
        - 🎯 **High Accuracy:** 89.10% validated  
        - 🌍 **Accessible:** Works on any device  
        - 🏥 **Aids Radiologists:** Reduces workload
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <p style="font-size: 0.9rem; color: #666;">
            Developed with ❤️ by<br>
            <b>KalyanaSundar</b><br>
            <span style="font-size: 0.8rem; color: #888;">AI Engineer</span>
        </p>
        <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
            <span class="badge">Python</span>
            <span class="badge">TensorFlow</span>
            <span class="badge">Streamlit</span>
            <span class="badge">DL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Main Content Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

# -----------------------------
# Left Column - Upload Section
# -----------------------------
with col1:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.8); border-radius: 20px; padding: 25px; backdrop-filter: blur(10px);">
        <h3 style="color: #333;">📤 Upload Chest X-Ray</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image (JPG or PNG format)"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(
            image,
            caption="🖼️ Uploaded Chest X-ray",
            use_container_width=True
        )
        
        # Display image info
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📐 Dimensions", f"{image.size[0]}×{image.size[1]}")
        with col_b:
            file_size = uploaded_file.size / 1024
            st.metric("📦 Size", f"{file_size:.1f} KB")

# -----------------------------
# Right Column - Prediction & Results
# -----------------------------
with col2:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.8); border-radius: 20px; padding: 25px; backdrop-filter: blur(10px);">
        <h3 style="color: #333;">🔬 Prediction Result</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # -----------------------------
    # Load Model
    # -----------------------------
    MODEL_PATH = os.path.join("model", "MobileNetV2_Tuned_Final.keras")
    
    @st.cache_resource
    def load_model():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except:
            st.error("⚠️ Model not found. Please ensure the model file exists.")
            return None
    
    model = load_model()
    
    # -----------------------------
    # Predict Button & Logic
    # -----------------------------
    if uploaded_file is not None and model is not None:
        if st.button("🚀 Predict", use_container_width=True):
            with st.spinner("🧠 Analyzing the X-ray image..."):
                time.sleep(0.5)  # Simulate processing
                
                # Preprocess image
                img = image.resize((224, 224))
                img = np.array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Predict
                prediction = model.predict(img)
                probability = float(prediction[0][0])
                
                # Determine result
                if probability >= 0.5:
                    predicted_class = "PNEUMONIA"
                    confidence = probability * 100
                    result_type = "error"
                else:
                    predicted_class = "NORMAL"
                    confidence = (1 - probability) * 100
                    result_type = "success"
                
                # Display results with animation
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Animated result card
                if result_type == "success":
                    st.markdown(f"""
                    <div class="success-card">
                        <h2 style="color: #1a6538; margin: 0;">✅ {predicted_class}</h2>
                        <p style="font-size: 1.1rem; margin: 10px 0 0 0;">
                            Confidence: <b>{confidence:.2f}%</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-card">
                        <h2 style="color: #7a1a1a; margin: 0;">⚠️ {predicted_class}</h2>
                        <p style="font-size: 1.1rem; margin: 10px 0 0 0;">
                            Confidence: <b>{confidence:.2f}%</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence progress bar
                st.markdown("#### Confidence Level")
                st.progress(min(int(confidence), 100))
                
                # Additional metrics
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("🎯 Confidence", f"{confidence:.2f}%")
                with col_y:
                    st.metric("📊 Raw Output", f"{probability:.4f}")
                
                # Interpretation
                st.markdown("#### 📖 Interpretation")
                if confidence > 90:
                    st.success("✅ **High Confidence:** This prediction is very reliable.")
                elif confidence > 70:
                    st.warning("⚠️ **Moderate Confidence:** Consider consulting a radiologist.")
                else:
                    st.error("❌ **Low Confidence:** Please consult a medical professional.")
                
                # Disclaimer
                st.markdown("""
                <div style="background: #f0f0f0; border-radius: 10px; padding: 15px; margin-top: 10px; font-size: 0.85rem;">
                    <b>⚠️ Medical Disclaimer:</b><br>
                    This tool is for educational and research purposes only. 
                    Always consult a qualified healthcare professional for medical diagnosis.
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# Bottom Section - Features & Info
# -----------------------------
st.markdown("---")

# Feature Cards
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    st.markdown("""
    <div class="float-card" style="text-align: center; background: linear-gradient(135deg, #667eea22, #764ba222);">
        <h1>⚡</h1>
        <h4>Rapid Analysis</h4>
        <p style="font-size: 0.9rem; color: #555;">Results in seconds</p>
    </div>
    """, unsafe_allow_html=True)

with col_f2:
    st.markdown("""
    <div class="float-card" style="text-align: center; background: linear-gradient(135deg, #f093fb22, #f5576c22);">
        <h1>🎯</h1>
        <h4>89.10% Accuracy</h4>
        <p style="font-size: 0.9rem; color: #555;">Validated performance</p>
    </div>
    """, unsafe_allow_html=True)

with col_f3:
    st.markdown("""
    <div class="float-card" style="text-align: center; background: linear-gradient(135deg, #84fab022, #8fd3f422);">
        <h1>🖼️</h1>
        <h4>Transfer Learning</h4>
        <p style="font-size: 0.9rem; color: #555;">MobileNetV2 architecture</p>
    </div>
    """, unsafe_allow_html=True)

with col_f4:
    st.markdown("""
    <div class="float-card" style="text-align: center; background: linear-gradient(135deg, #a18cd122, #fbc2eb22);">
        <h1>🌐</h1>
        <h4>Web Accessible</h4>
        <p style="font-size: 0.9rem; color: #555;">Any device, anywhere</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <p style="font-size: 0.9rem; color: #666;">
        <b>🫁 Pneumonia Detection System</b> · PRAICP-1012 · 
        Developed by <b>KalyanaSundar</b> · AI Engineer
    </p>
    <p style="font-size: 0.8rem; color: #999; margin-top: 5px;">
        Powered by TensorFlow & Streamlit · Fine-tuned MobileNetV2
    </p>
</div>
""", unsafe_allow_html=True)