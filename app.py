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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #f4fff7 0%, #ffffff 45%, #f7fffa 100%);
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 25px 15px 20px 15px;
    }

    .hero-icon {
        font-size: 55px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #176b3a;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #557064;
        max-width: 650px;
        margin: auto;
        line-height: 1.6;
    }

    /* Section titles */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #176b3a;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Result card */
    .result-card {
        background: white;
        border: 1px solid #dcefe2;
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(23, 107, 58, 0.08);
        text-align: center;
    }

    .result-label {
        color: #6b7f73;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .result-name {
        color: #176b3a;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .confidence-text {
        color: #35483d;
        font-size: 18px;
        font-weight: 600;
    }

    /* Info cards */
    .info-card {
        background: white;
        border: 1px solid #dcefe2;
        border-radius: 18px;
        padding: 20px;
        margin-top: 12px;
        box-shadow: 0 5px 18px rgba(23, 107, 58, 0.06);
    }

    .info-card h4 {
        color: #176b3a;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #52645a;
        line-height: 1.6;
    }

    /* Tip card */
    .tip-card {
        background: #ecfff2;
        border-left: 5px solid #2e9d59;
        border-radius: 14px;
        padding: 17px;
        margin-top: 15px;
        color: #28583a;
    }

    /* Category card */
    .category-row {
        background: white;
        border: 1px solid #e2eee6;
        border-radius: 13px;
        padding: 13px 16px;
        margin-bottom: 10px;
    }

    .category-name {
        font-weight: 650;
        color: #294638;
    }

    .category-percent {
        float: right;
        font-weight: 700;
        color: #176b3a;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #a8d8b8;
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 5px 18px rgba(23, 107, 58, 0.05);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 700;
        border: none;
        background: #238b4d;
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #176b3a;
        color: white;
        transform: translateY(-1px);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718078;
        font-size: 14px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #dce9df;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_waste_model():
    return load_model("model/waste_classifier.keras")


model = load_waste_model()


# =========================
# CLASS NAMES
# =========================
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
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">

    <div class="hero-icon">♻️</div>

    <div class="hero-title">
        Smart Waste Classifier
    </div>

    <div class="hero-subtitle">
        Upload an image of waste and let our CNN-based AI model
        identify its category and provide smart recycling guidance.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================
# HOW IT WORKS
# =========================
with st.expander("💡 How does it work?"):

    st.markdown("""
    **1️⃣ Upload**  
    Choose a clear image of a waste item.

    **2️⃣ Analyze**  
    Our CNN model analyzes the image.

    **3️⃣ Classify**  
    The system predicts the waste category.

    **4️⃣ Recycle**  
    Get disposal and recycling recommendations.
    """)


# =========================
# IMAGE UPLOAD
# =========================
st.markdown(
    '<div class="section-title">📸 Upload Your Waste Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================
# IMAGE PREVIEW
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your uploaded image",
        use_container_width=True
    )

    st.markdown("")

    analyze = st.button(
        "🔍 Analyze Waste",
        type="primary"
    )

    # =========================
    # PREDICTION
    # =========================
    if analyze:

        with st.spinner("🤖 AI is analyzing your image..."):

            resized_image = image.resize((128, 128))

            image_array = np.array(resized_image)

            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            prediction = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                prediction[0][predicted_index]
            ) * 100

        # =========================
        # RESULT
        # =========================
        st.markdown("""
        <div class="section-title">
            🎯 AI Prediction
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                Detected Waste Category
            </div>

            <div class="result-name">
                {predicted_class.title()}
            </div>

            <div class="confidence-text">
                Confidence: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(
            min(confidence / 100, 1.0)
        )


        # =========================
        # CATEGORY PERCENTAGES
        # =========================
        st.markdown("""
        <div class="section-title">
            📊 AI Confidence Breakdown
        </div>
        """, unsafe_allow_html=True)

        for i, name in enumerate(class_names):

            percent = float(
                prediction[0][i]
            ) * 100

            st.markdown(f"""
            <div class="category-row">

                <span class="category-name">
                    {name.title()}
                </span>

                <span class="category-percent">
                    {percent:.2f}%
                </span>

            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(percent / 100, 1.0)
            )


        # =========================
        # RECYCLING DETAILS
        # =========================
        info = recycling_info.get(
            predicted_class.lower(),
            {
                "title": predicted_class.title(),
                "method": "Dispose of this waste according to your local waste management guidelines.",
                "bin": "Follow your local waste collection rules.",
                "tip": "Separate recyclable materials from general waste whenever possible."
            }
        )

        st.markdown("""
        <div class="section-title">
            ♻️ Smart Recycling Guide
        </div>
        """, unsafe_allow_html=True)


        # How to recycle
        st.markdown(f"""
        <div class="info-card">

            <h4>🛠️ How to Recycle</h4>

            <p>
                {info["method"]}
            </p>

        </div>
        """, unsafe_allow_html=True)


        # Where to dispose
        st.markdown(f"""
        <div class="info-card">

            <h4>🗑️ Where to Dispose</h4>

            <p>
                {info["bin"]}
            </p>

        </div>
        """, unsafe_allow_html=True)


        # Recycling tip
        st.markdown(f"""
        <div class="tip-card">

            💡 <strong>Recycling Tip:</strong><br><br>

            {info["tip"]}

        </div>
        """, unsafe_allow_html=True)


        # =========================
        # ANOTHER IMAGE
        # =========================
        st.markdown("")
        st.info(
            "💚 Want to classify another item? "
            "Upload a new image above."
        )


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">

    ♻️ <strong>Smart Waste Classifier</strong><br>

    AI-powered waste classification for a cleaner and greener future 🌱

</div>
""", unsafe_allow_html=True)