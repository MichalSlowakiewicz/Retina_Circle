import streamlit as st
import PIL.Image
import PIL.ImageDraw
import json
import os
from pathlib import Path
import numpy as np

from src.algorithms.gradient import find_circle_gradient as find_circle_A
from src.algorithms.hough import find_circle_B

st.set_page_config(page_title="Eye Circle Detection", page_icon="👁️", layout="wide")

st.title("👁️ Eye Circle Detection: Interactive Viewer")
st.markdown("Upload a fundus image or pick one from the dataset to visualize the detection algorithms in action.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")

# Algorithm selection
algo_choice = st.sidebar.radio("Select Algorithm:", ["Algorithm A (Gradient Voting)", "Algorithm B (Optimized Hough)"])

# Load image
img_dir = Path("data/circle/val")
uploaded_file = st.sidebar.file_uploader("Upload an Image (jpg, png, webp)", type=["jpg", "png", "webp", "jpeg"])

selected_img_path = None
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    st.sidebar.success("Image uploaded successfully!")
else:
    # If no upload, try to load from local dataset
    if img_dir.exists():
        difficulties = [d for d in os.listdir(img_dir) if os.path.isdir(img_dir / d)]
        if difficulties:
            selected_diff = st.sidebar.selectbox("Select Difficulty Level:", sorted(difficulties))
            diff_dir = img_dir / selected_diff
            images = [f for f in os.listdir(diff_dir) if f.endswith(('.webp', '.jpg', '.png'))]
            if images:
                selected_img_name = st.sidebar.selectbox("Select Image:", sorted(images))
                selected_img_path = diff_dir / selected_img_name
                image = PIL.Image.open(selected_img_path).convert("RGB")
            else:
                st.sidebar.warning("No images found in the selected difficulty folder.")
                image = None
        else:
            st.sidebar.warning("No difficulty folders found in data/circle/val")
            image = None
    else:
        st.sidebar.warning("Local dataset not found at data/circle/val. Please upload an image.")
        image = None

# --- MAIN DISPLAY ---
if image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Algorithm Result")
        
        with st.spinner("Processing image..."):
            if algo_choice == "Algorithm A (Gradient Voting)":
                cy, cx, r = find_circle_A(image)
                color = "#f5c2e7"  # Catppuccin Pink
            else:
                cy, cx, r = find_circle_B(image)
                color = "#89b4fa"  # Catppuccin Blue
                
        # Draw on image
        result_image = image.copy()
        draw = PIL.ImageDraw.Draw(result_image)
        
        # Draw bounding circle
        draw.circle(xy=(cx, cy), radius=r, outline=color, width=5)
        # Draw center point
        draw.circle(xy=(cx, cy), radius=10, fill=color)
        
        st.image(result_image, use_container_width=True)
        
        st.success(f"**Detected Coordinates:** Center (X: {cx}, Y: {cy}) | Radius: {r}px")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & OpenCV")
