import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.upload-card {
    background: rgba(255,255,255,0.82);
    border: 2px dashed #22C55E;
    border-radius: 24px;
    padding: 35px;
    text-align: center;
    max-width: 800px;
    margin: 20px auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.07);
}

.upload-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.upload-heading {
    font-size: 25px;
    font-weight: 750;
    color: #14532D;
}

.upload-description {
    color: #64748B;
    margin-top: 8px;
    font-size: 15px;
}

.navbar {
    background: rgba(255,255,255,0.80);
    padding: 18px 30px;
    border-radius: 18px;
    margin-bottom: 40px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.logo {
    font-size: 25px;
    font-weight: 800;
    color: #14532D;
}

.nav-text {
    text-align: right;
    color: #52606D;
    font-size: 15px;
}

.hero {
    text-align: center;
    padding: 30px 10px 45px 10px;
}

.hero-badge {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    padding: 8px 18px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 600;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #14532D;
    margin-top: 18px;
    margin-bottom: 10px;
}

.hero-description {
    font-size: 18px;
    color: #64748B;
    max-width: 700px;
    margin: auto;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# NAVBAR
# ==============================

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        '<div class="logo">♻️ EcoVision AI</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="nav-text">Home &nbsp;&nbsp; | &nbsp;&nbsp; About &nbsp;&nbsp; | &nbsp;&nbsp; How It Works</div>',
        unsafe_allow_html=True
    )


# ==============================
# HERO SECTION
# ==============================

st.markdown("""
<div class="hero">

<div class="hero-badge">
🤖 AI Powered Waste Classification
</div>

<div class="hero-title">
Smart Waste Classification
</div>

<div class="hero-description">
Upload an image of waste and let our deep learning model
identify its category in seconds.
</div>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        #d7ffb7
    );
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)


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
st.markdown("""
<h1 style="text-align:center;
color:#087f5b;
font-size:45px;">
♻️ Smart Waste Classifier
</h1>
<p style="text-align:center;
color:#455a64;
font-size:20px;">
AI-Powered Waste Classification
</p>
""", unsafe_allow_html=True)

st.write(
    "Upload an image of waste and the AI model "
    "will predict its category."
    
)
st.markdown("""
<div class="upload-card">
    <div class="upload-icon">📷</div>
    <div class="upload-heading">Upload Your Waste Image</div>
    <div class="upload-description">
        Choose a clear image of cardboard, glass, metal, paper, plastic or trash.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)
st.markdown("""
<h3 style="color:#087f5b;">
📸 Upload Your Waste Image
</h3>
<p style="color:#087f5b;">
Upload an image and let AI identify its category.
</p>
""", unsafe_allow_html=True)


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

    st.markdown("""
   <div style="
    background:linear-gradient(135deg,#dcedc8,#b2dfdb);
    padding:20px;
    border-radius:15px;
    border-left:6px solid #2e7d32;
    ">
    <h3 style="color:#1b5e20;">
    🌱 AI Prediction Result
    </h3>
    <p style="color:#37474f;">
    Your waste category will appear here.
    </p>
    </div>
""", unsafe_allow_html=True)

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