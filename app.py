import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition System")
st.write("Photo upload kariye aur AI use pehchan lega!")

# Model Load ho raha hai
model = YOLO('best.pt')

# Upload Button
file = st.file_uploader("Upload Traffic Sign Photo", type=['jpg', 'png', 'jpeg'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Aapki Photo', use_container_width=True)

    # AI Prediction
    with st.spinner('AI soch raha hai...'):
        results = model(img)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='AI Result', use_container_width=True)
        st.success("Pechan liya gaya!")
