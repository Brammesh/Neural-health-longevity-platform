import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile
import tensorflow as tf
import textwrap
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.saving import register_keras_serializable

# --- Helper to convert image to base64 for CSS styling ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# --- FIX: Register the Custom Layer for Keras 3 ---
# This decorator is required to prevent "Could not locate class" errors.
@register_keras_serializable(package='Custom', name='LSTM')
class CustomLSTM(LSTM):
    """
    A custom LSTM layer that ignores the 'time_major' argument
    found in old Keras models to prevent crashes in Keras 3.
    """
    @classmethod
    def from_config(cls, config):
        if 'time_major' in config:
            del config['time_major']
        if 'implementation' in config and config['implementation'] == 0:
             config['implementation'] = 1
        return super().from_config(config)

# --- Custom CSS with Background Image ---
bg_img_b64 = get_base64_image("futuristic_medical_bg.png")
if bg_img_b64:
    bg_css_rule = f"""
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Outfit', sans-serif;
        background-image: linear-gradient(rgba(10, 12, 22, 0.88), rgba(10, 12, 22, 0.88)), url("data:image/png;base64,{bg_img_b64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    """
else:
    bg_css_rule = """
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: radial-gradient(circle at top right, #0d0f1b, #040509) !important;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    {bg_css_rule}
    
    /* Sleek gradient text for main header */
    .hero-title {{
        background: linear-gradient(135deg, #ff1744 0%, #00e5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 2px 8px rgba(0, 229, 255, 0.2));
    }}
    
    /* Neon glassmorphism card styling */
    div[data-testid="stVerticalBlock"] > div {{
        border-radius: 16px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    /* Premium styling for result alert banners */
    .stAlert {{
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(10px);
    }}
    
    /* Customize Streamlit Buttons */
    .stButton button {{
        background: linear-gradient(135deg, #ff1744 0%, #d500f9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 23, 68, 0.6) !important;
    }}
    
    /* File uploader styling */
    div[data-testid="stFileUploader"] {{
        border: 2px dashed rgba(0, 229, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        background: rgba(0, 229, 255, 0.02) !important;
        transition: all 0.3s ease;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: #ff1744 !important;
        background: rgba(255, 23, 68, 0.02) !important;
    }}
    
    /* Sidebar styled glassmorphic and clean */
    section[data-testid="stSidebar"] {{
        background: rgba(11, 12, 20, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }}
    
    /* Glassmorphic input containers */
    div[data-baseweb="input"], div[data-baseweb="select"], .stSelectbox select {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Glowing card headings */
    h3 {{
        color: #00e5ff !important;
        font-weight: 600;
        margin-top: 1.5rem !important;
        border-left: 4px solid #ff1744;
        padding-left: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Model Loading with Caching ---
@st.cache_resource
def load_heart_beat_model():
    return load_model(
        "heart_sounds.h5", 
        custom_objects={
            'LSTM': CustomLSTM, 
            'Bidirectional': Bidirectional
        }
    )

@st.cache_resource
def load_stress_model():
    return joblib.load("stress_model.pkl")

# Load models immediately
try:
    model = load_heart_beat_model()
    stress_model = load_stress_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Mapping of stress class to descriptive text
stress_mapping = {0: "Low", 1: "Moderate", 2: "High"}

# --- Preprocessing Function (Robust Version) ---
def preprocess_audio(uploaded_file):
    # Safe loading using tempfile to prevent librosa/Streamlit crashes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        audio, sr = librosa.load(tmp_file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=25)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs.reshape(1, 25, 1)
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# --- Risk Calculation ---
def calculate_heart_failure_risk(murmur_prob, age):
    base_risk = murmur_prob * 0.7
    age_risk = max(0, (age - 30) * 0.1)
    total_risk = base_risk + age_risk
    total_risk = max(0, min(100, total_risk))
    return total_risk

ANTIGRAVITY_JS = """<script>
(function(){
    const parentWin = window.parent;
    if (!parentWin || parentWin.__antigravity_initialized) return;
    parentWin.__antigravity_initialized = true;

    const parentDoc = parentWin.document;

    const style = parentDoc.createElement('style');
    style.textContent = '@keyframes heartbeat-trail{0%{transform:translate(-50%,-50%) scale(0);opacity:0}15%{transform:translate(-50%,-50%) scale(1.4);opacity:0.95}30%{transform:translate(-50%,-50%) scale(0.9);opacity:0.85}45%{transform:translate(-50%,-50%) scale(1.2);opacity:0.8}100%{transform:translate(-50%,-50%) scale(0);opacity:0}}.heart-trail-element{position:fixed;pointer-events:none;z-index:999998;animation:heartbeat-trail 0.8s cubic-bezier(0.25,1,0.5,1) forwards;width:28px;height:28px}';
    parentDoc.head.appendChild(style);

    let lastX = 0, lastY = 0;
    parentWin.addEventListener('mousemove', (e) => {
        const dist = Math.hypot(e.clientX - lastX, e.clientY - lastY);
        if (dist > 25) {
            draw(e.clientX, e.clientY);
            lastX = e.clientX;
            lastY = e.clientY;
        }
    });

    function draw(x, y) {
        const div = parentDoc.createElement('div');
        div.className = 'heart-trail-element';
        div.style.left = x + 'px';
        div.style.top = y + 'px';
        const colors = ['#ff1744','#00e5ff','#d500f9'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        div.innerHTML = '<svg viewBox="0 0 24 24" fill="' + color + '" width="100%" height="100%" style="filter:drop-shadow(0 2px 8px ' + color + ')"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>';
        parentDoc.body.appendChild(div);
        const activeTrails = parentDoc.querySelectorAll('.heart-trail-element');
        if (activeTrails.length > 25) {
            activeTrails[0].remove();
        } else {
            setTimeout(() => { div.remove(); }, 800);
        }
    }

    const panel = parentDoc.createElement('div');
    panel.style.position = 'fixed';
    panel.style.bottom = '20px';
    panel.style.right = '20px';
    panel.style.zIndex = '999999';
    panel.style.background = 'rgba(11,12,20,0.9)';
    panel.style.color = '#ff1744';
    panel.style.border = '2px solid #ff1744';
    panel.style.borderRadius = '30px';
    panel.style.padding = '12px 24px';
    panel.style.boxShadow = '0 8px 32px 0 rgba(255,23,68,0.4)';
    panel.style.backdropFilter = 'blur(10px)';
    panel.style.fontFamily = 'Outfit,sans-serif';
    panel.style.fontSize = '14px';
    panel.style.fontWeight = 'bold';
    panel.style.cursor = 'pointer';
    panel.style.transition = 'all 0.3s ease';
    panel.style.userSelect = 'none';
    panel.innerHTML = '🌌 Activate Antigravity';
    parentDoc.body.appendChild(panel);

    let active = false;
    let elements = [];
    let mouse = {x:0, y:0, px:0, py:0, vx:0, vy:0, down:false, target:null};

    parentWin.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        mouse.vx = mouse.x - mouse.px;
        mouse.vy = mouse.y - mouse.py;
        mouse.px = mouse.x;
        mouse.py = mouse.y;
        if (mouse.down && mouse.target) {
            mouse.target.dx += mouse.vx;
            mouse.target.dy += mouse.vy;
            mouse.target.vx = mouse.vx;
            mouse.target.vy = mouse.vy;
        }
    });

    parentWin.addEventListener('mousedown', (e) => {
        if (!active) return;
        if (e.target === panel || panel.contains(e.target)) return;
        mouse.down = true;
        const path = e.composedPath();
        for (let el of path) {
            if (el.classList && (el.classList.contains('element-container') || el.tagName === 'H1' || el.tagName === 'H2' || el.tagName === 'P' || el.tagName === 'IMG' || el.classList.contains('stButton') || el.classList.contains('stAudio'))) {
                const match = elements.find(item => item.dom === el);
                if (match) {
                    mouse.target = match;
                    break;
                }
            }
        }
    });

    parentWin.addEventListener('mouseup', () => {
        mouse.down = false;
        mouse.target = null;
    });

    panel.addEventListener('click', () => {
        active = !active;
        if (active) {
            panel.innerHTML = '🌎 Restore Gravity';
            panel.style.color = '#00e5ff';
            panel.style.borderColor = '#00e5ff';
            panel.style.boxShadow = '0 8px 32px 0 rgba(0,229,255,0.4)';
            initializePhysics();
        } else {
            panel.innerHTML = '🌌 Activate Antigravity';
            panel.style.color = '#ff1744';
            panel.style.borderColor = '#ff1744';
            panel.style.boxShadow = '0 8px 32px 0 rgba(255,23,68,0.4)';
            resetPhysics();
        }
    });

    function initializePhysics() {
        elements = [];
        const targets = parentDoc.querySelectorAll('.element-container,h1,h2,img,.stButton,.stAudio,.stNumberInput,.stSelectbox,.stFileUploader');
        const seen = new Set();
        targets.forEach(el => {
            if (el === panel) return;
            if (el.querySelector('.element-container')) return;
            if (seen.has(el)) return;
            seen.add(el);
            elements.push({dom:el, dx:0, dy:0, vx:(Math.random()-0.5)*6, vy:(Math.random()-0.5)*6-3, width:el.offsetWidth, height:el.offsetHeight});
            el.style.willChange = 'transform';
            el.style.transition = 'none';
        });
    }

    function resetPhysics() {
        elements.forEach(item => {
            item.dom.style.transform = '';
            item.dom.style.willChange = '';
        });
        elements = [];
    }

    function update() {
        requestAnimationFrame(update);
        if (!active) return;
        elements.forEach(item => {
            if (mouse.down && mouse.target === item) {
                item.dom.style.transform = 'translate3d(' + item.dx + 'px,' + item.dy + 'px,0px)';
                return;
            }
            item.vy -= 0.04;
            item.vx += (Math.random() - 0.5) * 0.2;
            item.vy += (Math.random() - 0.5) * 0.2;
            item.vx *= 0.97;
            item.vy *= 0.97;
            const rect = item.dom.getBoundingClientRect();
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const dx = cx - mouse.x;
            const dy = cy - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 180 && dist > 1) {
                const force = (180 - dist) * 0.12;
                item.vx += (dx / dist) * force;
                item.vy += (dy / dist) * force;
            }
            item.dx += item.vx;
            item.dy += item.vy;
            const buffer = 15;
            if (rect.left + item.vx < buffer) {
                item.vx = Math.abs(item.vx) * 0.85;
                item.dx += item.vx;
            }
            if (rect.right + item.vx > parentWin.innerWidth - buffer) {
                item.vx = -Math.abs(item.vx) * 0.85;
                item.dx += item.vx;
            }
            if (rect.top + item.vy < buffer) {
                item.vy = Math.abs(item.vy) * 0.85;
                item.dy += item.vy;
            }
            if (rect.bottom + item.vy > parentWin.innerHeight - buffer) {
                item.vy = -Math.abs(item.vy) * 0.85;
                item.dy += item.vy;
            }
            item.dom.style.transform = 'translate3d(' + item.dx + 'px,' + item.dy + 'px,0px)';
        });
    }
    update();
})();
</script>"""

# --- Main App ---
def main():
    # Styled hero header
    st.markdown('<h1 class="hero-title">❤️ Heart Failure & Stress Level Assessment</h1>', unsafe_allow_html=True)
    
    # Beautiful project hero banner image
    if os.path.exists("heart_stress_concept.png"):
        st.image("heart_stress_concept.png", use_container_width=True)
        
    st.write("""
    Welcome to the **Heart Failure Risk Assessment** app!  
    This app uses a **Bidirectional LSTM (BiLSTM)** model trained on heart sound data to classify heartbeats and estimate the risk of heart failure.  
    Upload a **WAV file** of a heartbeat, and we'll analyze it for you!  
    Additionally, provide your personal details below which our stress model uses to predict your stress level (Low, Moderate, High).  
    Combined recommendations (including dietary suggestions) are provided based on both outputs.
    """)
    
    # Inject JavaScript for Google Antigravity Easter Egg
    st.components.v1.html(ANTIGRAVITY_JS, height=0)

    # Sidebar with project and dataset info
    st.sidebar.title("ℹ️ About This Project")
    st.sidebar.write("""
    ### **Heart Sound Model Details** - **Model Architecture**: Bidirectional LSTM (BiLSTM)  
    - **Input Shape**: (25, 1)  
    - **Output Classes**: Artifact, Murmur, Normal  
    - **Training Dataset**: [PhysioNet CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/)  
    - **Purpose**: Classify heart sounds and estimate heart failure risk.  
    """)
    st.sidebar.write("""
    ### **Stress Model Details** - **Features**: anxiety_level, mental_health_history, depression, headache, sleep_quality, breathing_problem, living_conditions  
    - **Output**: Predicted Stress Level Class (0=Low, 1=Moderate, 2=High)  
    - **Purpose**: Estimate your stress level from input features.
    """)
    st.sidebar.write("""
    ### **How It Works** 1. Upload a WAV file of a heartbeat.  
    2. The app extracts MFCC features from the audio.  
    3. The BiLSTM model predicts the probabilities of each heart sound class.  
    4. The app calculates the risk of heart failure based on the murmur probability and your age.  
    5. Provide your details for stress prediction.  
    6. Combined recommendations are provided.
    """)

    # Input: Age for heart risk calculation
    age = st.number_input("👤 Enter your age:", min_value=1, max_value=120, value=30)

    # Input: Upload WAV file for heart sound analysis
    st.write("### 🎵 Upload Heartbeat Audio")
    uploaded_file = st.file_uploader("Upload a WAV file:", type=["wav"])

    risk_percentage = None  # in case no file is uploaded
    predicted_stress_class = None

    if uploaded_file is not None:
        try:
            st.audio(uploaded_file, format="audio/wav")
            # Using the safe preprocess function
            data = preprocess_audio(uploaded_file)
            predictions = model.predict(data)
            
            artifact_prob = predictions[0][0] * 100
            murmur_prob = predictions[0][1] * 100
            normal_prob = predictions[0][2] * 100

            st.write("### 📊 Heart Sound Classification Probabilities")
            st.write(f"- 🎛️ Artifact: {artifact_prob:.2f}%")
            st.write(f"- ❤️ Murmur: {murmur_prob:.2f}%")
            st.write(f"- ✅ Normal: {normal_prob:.2f}%")

            risk_percentage = calculate_heart_failure_risk(murmur_prob, age)
            st.write("### 🚨 Heart Failure Risk Assessment")
            st.write(f"**Risk of heart failure: {risk_percentage:.2f}%**")

            if risk_percentage >= 70:
                st.error("⚠️ Very high risk of heart failure. Seek immediate medical attention.")
            elif risk_percentage >= 40:
                st.warning("🔔 Moderate to high risk of heart failure. Consult a cardiologist.")
            elif risk_percentage >= 20:
                st.info("ℹ️ Low to moderate risk of heart failure. Monitor your health.")
            else:
                st.success("✅ Low risk of heart failure. No immediate concern.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    # --- Stress Prediction Section ---
    st.write("### 😰 Stress Level Prediction")
    st.write("Please provide the following details:")

    anxiety_level = st.number_input("Anxiety Level", min_value=0, max_value=100, value=13)
    mental_health_history = st.selectbox("Mental Health History?", options=["No", "Yes"], index=0)
    depression = st.number_input("Depression", min_value=0, max_value=100, value=10)
    headache = st.number_input("Headache", min_value=0, max_value=100, value=1)
    sleep_quality = st.number_input("Sleep Quality", min_value=0, max_value=10, value=1)
    breathing_problem = st.number_input("Breathing Problem", min_value=0, max_value=10, value=2)
    living_conditions = st.number_input("Living Conditions", min_value=0, max_value=10, value=2)

    mental_health_history_val = 1 if mental_health_history == "Yes" else 0

    X_stress = np.array([[anxiety_level, mental_health_history_val, depression, headache, sleep_quality, breathing_problem, living_conditions]], dtype=float)

    try:
        # Auto-predict stress (as in original code)
        predicted_stress_class = stress_model.predict(X_stress)[0]
        st.write(f"**Predicted Stress Level:** {stress_mapping.get(predicted_stress_class, 'Unknown')} (Class {predicted_stress_class})")
    except Exception as e:
        st.error(f"Error predicting stress level: {e}")
        predicted_stress_class = None

    # --- Combined Recommendation Section ---
    st.write("### 🔔 Combined Recommendation")
    recommendation = ""
    
    # Logic from original code
    if predicted_stress_class is not None and risk_percentage is not None:
        if predicted_stress_class == 2 and risk_percentage >= 70:
            recommendation += (
                "⚠️ **High Alert:** Your stress level is predicted as **High** and your heart failure risk is very high. "
                "Immediate medical consultation is strongly recommended. Adopt a heart-healthy diet rich in fruits, vegetables, "
                "whole grains, and lean proteins, while minimizing salt and saturated fats. Engage in stress reduction techniques such as meditation, yoga, or counseling."
            )
        elif predicted_stress_class == 2 and risk_percentage < 70:
            recommendation += (
                "⚠️ **High Stress Detected:** Your stress level is predicted as **High**, although your heart risk is moderate. "
                "Consider adopting stress management strategies including regular exercise, meditation, and proper sleep. "
                "A balanced diet featuring omega-3 rich foods (e.g., fish, nuts) and low in processed foods is advisable."
            )
        elif predicted_stress_class == 0 and risk_percentage >= 70:
            recommendation += (
                "⚠️ **Heart Risk Alert:** Despite a **Low** predicted stress level, your heart failure risk is high. "
                "It is important to consult a cardiologist promptly. In the meantime, focus on a heart-friendly diet that reduces salt and unhealthy fats "
                "and increases fruits, vegetables, and whole grains."
            )
        else:
            recommendation += (
                "✅ **Overall Stable:** Your predicted stress level and heart failure risk appear within acceptable ranges. "
                "Maintain a balanced lifestyle with regular physical activity, proper sleep, and a diet rich in antioxidants and anti-inflammatory foods."
            )
    else:
        recommendation = "Insufficient data for combined recommendation. Please ensure both heart sound and stress inputs are provided."
    
    st.write(recommendation)

if __name__ == "__main__":
    main()