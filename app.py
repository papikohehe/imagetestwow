# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# -----------------------------
# Utility Functions
# -----------------------------

def is_pure_color(img, std_threshold=15, unique_threshold=0.02):
    """
    Check if image is mostly one color (allowing small gradient or noise).
    """
    pixels = img.reshape(-1, img.shape[-1])
    stddev = np.std(pixels)

    # Quantize colors to reduce sensitivity
    quantized = (pixels // 16).astype(np.uint8)
    unique_colors = len(np.unique(quantized, axis=0))
    ratio_unique = unique_colors / len(pixels)

    return (stddev < std_threshold) and (ratio_unique < unique_threshold)


def has_object(img, edge_threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    return edge_pixels > edge_threshold


def classify_image(img):
    if has_object(img):
        return "Has object"
    elif is_pure_color(img):
        return "Pure color (or mostly uniform with gradient)"
    else:
        return "Uncertain"


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"❌ Could not load image from URL: {e}")
        return None

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Image Classifier", page_icon="🖼️")

st.title("🖼️ Lightweight Image Classifier")
st.write("Classify an image as **Pure Color** or **Has Object**.")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

# URL input
image_url = st.text_input("Or enter an image URL")

# Process Image
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif image_url:
    image = load_image_from_url(image_url)

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Classification
    result = classify_image(img_cv2)
    st.subheader(f"🔎 Result: {result}")

    # Debugging info
    with st.expander("Show Debugging Info"):
        pixels = img_cv2.reshape(-1, img_cv2.shape[-1])
        stddev = np.std(pixels)
        quantized = (pixels // 16).astype(np.uint8)
        unique_colors = len(np.unique(quantized, axis=0))
        ratio_unique = unique_colors / len(pixels)

        edge_pixels = np.sum(cv2.Canny(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY), 100, 200) > 0)

        st.write("Standard Deviation:", stddev)
        st.write("Unique Color Ratio:", ratio_unique)
        st.write("Edge Pixels:", edge_pixels)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenCV")
