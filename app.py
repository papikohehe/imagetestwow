# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Utility Functions
# -----------------------------

def is_pure_color(img, std_threshold=15, unique_threshold=0.02):
    """
    Check if image is mostly one color (allowing small gradient or noise).
    - std_threshold: lower = stricter, higher = allow small gradients
    - unique_threshold: ratio of unique colors to total pixels
    """
    pixels = img.reshape(-1, img.shape[-1])

    # Standard deviation across channels
    stddev = np.std(pixels)

    # Count unique colors (quantized to reduce sensitivity to tiny changes)
    quantized = (pixels // 16).astype(np.uint8)
    unique_colors = len(np.unique(quantized, axis=0))
    ratio_unique = unique_colors / len(pixels)

    return (stddev < std_threshold) and (ratio_unique < unique_threshold)


def is_blurry(img, threshold=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


def has_object(img, edge_threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    return edge_pixels > edge_threshold


def classify_image(img):
    if is_pure_color(img):
        return "Pure color (or mostly uniform with gradient)"
    elif is_blurry(img):
        return "Blurry"
    elif has_object(img):
        return "Has object"
    else:
        return "Uncertain"

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Image Classifier", page_icon="🖼️")

st.title("🖼️ Lightweight Image Classifier")
st.write("Classify an image as **Pure Color**, **Blurry**, or **Has Object**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

        lap_var = cv2.Laplacian(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        edge_pixels = np.sum(cv2.Canny(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY), 100, 200) > 0)

        st.write("Standard Deviation:", stddev)
        st.write("Unique Color Ratio:", ratio_unique)
        st.write("Laplacian Variance:", lap_var)
        st.write("Edge Pixels:", edge_pixels)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenCV")
