import streamlit as st
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model
import gdown
import os
from io import BytesIO

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


# =====================================
# LOAD TRAINED MODEL
# =====================================
@st.cache_resource
def load_waste_model():
    model_path = "model/waste_classifier.keras"

    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        gdown.download(
            id="1_O-l1dZNbD4QZub2-i6hjqGMv3lv6qZS",
            output=model_path,
            quiet=False
        )

    return load_model(model_path)

model = load_waste_model()

# =====================================
# CLASS NAMES
# =====================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# =====================================
# TITLE
# =====================================

st.title("♻️ Smart Waste Classifier")

st.write(
    "Upload an image of waste and the AI model "
    "will predict its category."
)


# IMAGE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(
        BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    # Show image
    st.image(image, caption="Uploaded Waste Image")

    # Resize image
    image = image.resize((128, 128))

    # Convert image to array
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)

    # Get predicted class
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    # Show result
    st.success(f"Predicted category: {predicted_class}")

    confidence = float(np.max(prediction[0])) * 100
    st.write(f"Confidence: {confidence:.2f}%")

    details = {
            "plastic": "Plastic waste - Recycle it.",
            "paper": "Paper waste - Recycle it.",
            "metal": "Metal waste - Send it for recycling.",
            "glass": "Glass waste - Recycle it safely.",
            "cardboard": "Cardboard waste - Recycle it.",
            "trash": "General waste - Dispose of it properly."
        }

    st.info(details.get(predicted_class.lower(), "No extra details available."))