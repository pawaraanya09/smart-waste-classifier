import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

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
        "method": "Boxes ko flatten karo aur saaf aur sukha rakho.",
        "bin": "Dry waste bin",
        "tip": "Boxes ko dobara use kar sakte ho."
    },
    "glass": {
        "title": "Glass Waste",
        "method": "Glass bottles aur jars ko alag collect karo.",
        "bin": "Glass recycling collection",
        "tip": "Broken glass ko carefully handle karo."
    },
    "metal": {
        "title": "Metal Waste",
        "method": "Metal cans ko saaf karke recycling ke liye alag rakho.",
        "bin": "Dry waste bin",
        "tip": "Metal containers ko reuse bhi kar sakte ho."
    },
    "paper": {
        "title": "Paper Waste",
        "method": "Paper ko saaf aur sukha rakho aur recycling ke liye bhejo.",
        "bin": "Dry waste bin",
        "tip": "Paper ko recycle karne se pehle reuse karo."
    },
    "plastic": {
        "title": "Plastic Waste",
        "method": "Plastic ko saaf aur sukha karke recyclable plastic alag rakho.",
        "bin": "Dry waste bin",
        "tip": "Single-use plastic ka use kam karo."
    },
    "trash": {
        "title": "General Waste",
        "method": "Waste ko local waste collection rules ke according dispose karo.",
        "bin": "General waste bin",
        "tip": "Recyclable items ko general waste se alag rakho."
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