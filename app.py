import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("EcoSort AI")
st.write("Smart Waste Classification System")

model = load_model("waste_model.keras")

class_names = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((128, 128))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    index = np.argmax(prediction)

    st.success("Predicted Category: " + class_names[index])
    st.write("Confidence:", round(prediction[index] * 100, 2), "%")