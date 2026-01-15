import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# 1. Page Setup
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image, and the AI will identify it.")

# 2. Load Model (Cached for speed)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Analyzing...")

    # 4. AI Prediction
    try:
        results = model(image)
        
        # Get the result image with boxes
        res_plotted = results[0].plot()
        
        # COLOR FIX: Convert BGR (Blue) to RGB (Red)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 5. Display Result
        st.success("✅ Detection Complete!")
        st.image(res_rgb, caption="AI Detection Result", use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
