import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Signature Visualizer")

st.markdown("""
Upload one or more `.npy` files containing preprocessed signature images.
Each file should contain a normalized 2D NumPy array (values in [0, 1]) representing a signature.
""")

uploaded_files = st.file_uploader("Choose .npy files", type=["npy"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read file as bytes then load with numpy
        bytes_data = uploaded_file.read()
        # Using BytesIO to simulate a file for np.load
        npy_file = BytesIO(bytes_data)
        try:
            img = np.load(npy_file)
            st.write(f"### {uploaded_file.name}")
            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
