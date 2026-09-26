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

    # Show confidence
confidence = prediction[0][predicted_index] * 100
st.write(f"Confidence: {confidence:.2f}%")

# Recycling information
recycling_info = {
    "paper": {
        "title": "Paper Waste",
        "method": "Keep paper clean and dry. Remove plastic covers and send it for recycling.",
        "bin": "Blue dry waste bin",
        "tip": "Reuse paper before recycling."
    },
    "plastic": {
        "title": "Plastic Waste",
        "method": "Clean the plastic item, dry it, and send recyclable plastic to a recycling center.",
        "bin": "Blue dry waste bin",
        "tip": "Avoid single-use plastic."
    },
    "metal": {
        "title": "Metal Waste",
        "method": "Clean metal cans and separate them for metal recycling.",
        "bin": "Dry waste bin",
        "tip": "Reuse metal containers whenever possible."
    },
    "glass": {
        "title": "Glass Waste",
        "method": "Separate glass bottles and jars carefully and send them for glass recycling.",
        "bin": "Glass recycling collection",
        "tip": "Handle broken glass carefully."
    },
    "cardboard": {
        "title": "Cardboard Waste",
        "method": "Flatten cardboard boxes and keep them dry for recycling.",
        "bin": "Blue dry waste bin",
        "tip": "Reuse boxes before recycling."
    }
}

category = str(predicted_class).lower().strip()

if category in recycling_info:
    info = recycling_info[category]

    st.subheader("♻️ Recycling Details")
    st.write("**Waste Type:**", info["title"])
    st.write("**How to Recycle:**", info["method"])
    st.write("**Where to Dispose:**", info["bin"])
    st.info("💡 " + info["tip"])