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