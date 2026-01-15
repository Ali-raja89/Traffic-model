import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2  # Ye sirf color thik karne ke liye hai

# 1. Page Configuration
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a photo and the AI will identify it!")

# 2. Loading the Model
# Cache ka use kiya taaki bar-bar load na ho
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. Upload Button
file = st.file_uploader("Upload Traffic Sign Photo", type=['jpg', 'png', 'jpeg'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Photo', use_container_width=True)

    # 4. AI Prediction
    with st.spinner('AI is analyzing...'):
        results = model(img)
        
        # Result ko plot karna
        res_plotted = results[0].plot()
        
        # Color Fix: BGR se RGB (Taaki photo neeli na dikhe)
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Displaying the result image with detection boxes
        st.image(res_plotted, caption='AI Result', use_container_width=True)
        
        # Success message
        st.success("Identification Complete!")
