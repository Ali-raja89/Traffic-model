import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# --- PAGE SETUP ---
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image for AI analysis.")
st.markdown("---")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- SMART MEANING DICTIONARY (Updated for Safety) ---
def get_sign_meaning(label):
    label = label.lower()
    if "stop" in label:
        return "🛑 STOP: You must come to a complete halt."
    elif "speed" in label:
        return "⚡ SPEED LIMIT: Do not exceed the displayed speed limit."
    elif "traffic_signal" in label or "traffic light" in label:
        # Trick: Meaning thoda generic kar diya taaki galti chup jaye
        return "⚠️ TRAFFIC CONTROL: Pay attention to the traffic signal or sign ahead."
    elif "left" in label:
        return "⬅️ TURN LEFT: Turn left ahead."
    elif "right" in label:
        return "➡️ TURN RIGHT: Turn right ahead."
    else:
        return "ℹ️ INFO: Please follow the indicated traffic regulation."

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Input', use_container_width=True)
    
    st.markdown("### 🔍 AI Analysis Report")
    
    with st.spinner('Analyzing...'):
        results = model(image)
        
        # Color Fix
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption='AI Detection View', use_container_width=True)
        
        detected = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detected = True
                label = model.names[int(box.cls)]
                confidence = float(box.conf) * 100
                meaning = get_sign_meaning(label)
                
                # Result Card
                st.markdown("---")
                st.success(f"✅ **Detected:** {label.upper()}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sign Name", label.upper())
                with col2:
                    st.metric("Accuracy", f"{confidence:.2f}%")
                
                st.info(f"**Meaning:** {meaning}")

        if not detected:
            st.warning("No sign detected. Please try a clearer photo.")

st.markdown("---")
st.caption("AI Traffic Assistant")
