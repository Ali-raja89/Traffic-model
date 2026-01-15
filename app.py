import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Page Configuration
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to detect it.")

# 2. Load Model
# Using cache to prevent reloading on every interaction
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Check if 'best.pt' exists. {e}")

# 3. Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # A. Display Original Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Analyzing...")

    try:
        # 4. PRE-PROCESSING
        # Convert PIL Image to NumPy Array (RGB)
        img_array = np.array(image)
        
        # YOLO works best with this format. 
        results = model(img_array)

        # 5. Result Handling
        # Get the plotted image (BGR format from YOLO)
        res_plotted = results[0].plot()
        
        # Convert BGR back to RGB for Streamlit display
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 6. Display Result
        st.success("Analysis Complete!")
        st.image(res_rgb, caption="AI Detection Result", use_container_width=True)

        # Print detected class name below
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                st.info(f"Detected: {class_name.upper()} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
