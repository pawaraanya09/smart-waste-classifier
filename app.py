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


# =====================================
# IMAGE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📷 Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)


# Resize image
image = image.resize((128, 128))

# Convert image to NumPy array
image_array = np.array(image)

# Normalize pixel values
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    # Open image
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")

    # Show image
    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    # Make prediction
    prediction = model.predict(image_array)
    st.write("Raw Prediction:", prediction[0])

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    # Resize image
    image = image.resize((128, 128))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Find class with highest probability
    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100


    # =================================
    # SHOW RESULT
    # =================================

    st.subheader("🔍 Prediction Result")

    st.success(
        f"♻️ Waste Type: {predicted_class.upper()}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # =================================
    # SHOW ALL PROBABILITIES
    # =================================

    st.subheader("📊 Prediction Probabilities")

    for i in range(len(class_names)):

        probability = prediction[0][i]

        st.write(
            f"{class_names[i].title()}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # =================================
    # RECYCLING SUGGESTION
    # =================================

    suggestions = {

        "cardboard":
            "📦 Flatten cardboard and place it with recyclable paper/cardboard.",

        "glass":
            "🍾 Separate glass items and place them in the appropriate glass recycling stream.",

        "metal":
            "🥫 Empty and clean metal containers before recycling.",

        "paper":
            "📄 Keep paper dry and place it with recyclable paper.",

        "plastic":
            "🧴 Empty and clean plastic containers before recycling.",

        "trash":
            "🗑️ Dispose of this item according to your local waste-management rules."
    }


    st.subheader("💡 Suggested Action")

    st.warning(
        suggestions[predicted_class]
    )


# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Smart Waste Classification using "
    "Deep Learning & CNN"
)