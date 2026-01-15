import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Traffic AI", page_icon="🚦")

# 2. Professional Header
st.title("🚦 Traffic Sign Recognition System")
st.markdown("---")
st.write("Welcome! Please upload an image of a traffic sign for real-time AI identification.")

# 3. Load the AI Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Image Upload Section
uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Traffic Sign', use_container_width=True)
    
    with st.spinner('AI is analyzing the image... Please wait.'):
        # AI Prediction
        results = model(image)
        
        st.markdown("### **Detection Results:**")
        
        # Checking if any sign is detected
        detected = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detected = True
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                
                # Show success message in English
                st.success(f"**Sign Detected:** {label.upper()}")
                st.info(f"**Confidence Score:** {confidence:.2f}")

        if not detected:
            st.warning("No traffic sign detected in this image. Please try another one.")

# 5. Footer
st.markdown("---")
st.caption("Developed with YOLOv8 & Streamlit")
