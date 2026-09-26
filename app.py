import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

import streamlit as st

st.set_page_config(
    page_title=“Smart Waste Classifier”,
    page_icon=“♻️”,
    layout=“wide”
)
st.markdown(“””
<style>

body {
    background-color: #F1F8F4;
}

/* Main heading */
.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #087F5B;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #52796F;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Upload card */
.upload-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    border: 2px solid #D8F3DC;
}

/* Result card */
.result-card {
    background: #E9FBEF;
    padding: 25px;
    border-radius: 20px;
    border-left: 8px solid #2D6A4F;
    margin-top: 25px;
}

/* Info cards */
.info-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.07);
}

</style>
“””, unsafe_allow_html=True)

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_waste_model():
    return load_model("model/waste_classifier.keras")

model = load_waste_model()

# IMPORTANT:
# Class names ka order train.py ke order se
# bilkul same hona chahiye.
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# =========================
# RECYCLING INFORMATION
# =========================
recycling_info = {

    "cardboard": {
        "title": "Cardboard Waste",
        "method": "Flatten the boxes and keep them clean and dry.",
        "bin": "Dry Waste Bin",
        "tip": "You can reuse cardboard boxes before recycling them."
    },

    "glass": {
        "title": "Glass Waste",
        "method": "Collect glass bottles and jars separately for recycling.",
        "bin": "Glass Recycling Collection",
        "tip": "Handle broken glass carefully and keep it separate from other waste."
    },

    "metal": {
        "title": "Metal Waste",
        "method": "Clean metal cans and keep them separate for recycling.",
        "bin": "Dry Waste Bin",
        "tip": "Metal containers can also be reused before recycling."
    },

    "paper": {
        "title": "Paper Waste",
        "method": "Keep paper clean and dry and send it for recycling.",
        "bin": "Dry Waste Bin",
        "tip": "Reuse paper whenever possible before recycling it."
    },

    "plastic": {
        "title": "Plastic Waste",
        "method": "Clean and separate recyclable plastic items before disposal.",
        "bin": "Dry Waste Bin",
        "tip": "Reduce the use of single-use plastics and choose reusable alternatives."
    },

    "trash": {
        "title": "General Waste",
        "method": "Dispose of general waste according to your local waste collection guidelines.",
        "bin": "General Waste Bin",
        "tip": "Keep recyclable items separate from general waste."
    }
}

# =========================
# WEBSITE TITLE
# =========================
st.title("♻️ Smart Waste Classifier")

st.write(
    "Upload an image of waste and the AI model "
    "will predict its category."
)

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    # Resize image
    image = image.resize((128, 128))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixels
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = float(prediction[0][predicted_index]) * 100

    # =========================
    # SHOW PREDICTION
    # =========================
    st.success(f"Predicted Category: {predicted_class.title()}")

    st.subheader("🎯 Prediction Confidence")
    st.write(f"{confidence:.2f}%")
    st.progress(min(confidence / 100, 1.0))

    # =========================
    # CATEGORY PERCENTAGES
    # =========================
    st.subheader("📊 Waste Category Percentages")

    for i, name in enumerate(class_names):
        percent = float(prediction[0][i]) * 100

        st.write(f"**{name.title()}: {percent:.2f}%**")
        st.progress(min(percent / 100, 1.0))

    # =========================
    # RECYCLING DETAILS
    # =========================
    st.subheader("♻️ Recycling Details")

    info = recycling_info.get(
        predicted_class.lower(),
        {
            "title": predicted_class.title(),
            "method": "Is waste ko local recycling guidelines ke according dispose karo.",
            "bin": "Local waste collection rules follow karo.",
            "tip": "Waste ko alag-alag categories mein sort karo."
        }
    )

    st.info(f"**Waste Type:** {info['title']}")

    st.write("### 🛠️ How to Recycle")
    st.write(info["method"])

    st.write("### 🗑️ Where to Dispose")
    st.write(info["bin"])

    st.success(f"💡 Recycling Tip: {info['tip']}")