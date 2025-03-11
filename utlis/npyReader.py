import streamlit as st
import numpy as np
import json

# App title
st.title("View and Copy `.npy` File Data")

# File uploader
uploaded_file = st.file_uploader("Upload a `.npy` file", type=["npy"])

if uploaded_file:
    try:
        # Load the .npy file
        data = np.load(uploaded_file)

        # Flatten the data
        flattened_data = data.flatten().tolist()

        # Create a JSON-like format for output
        output = {"landmarks": flattened_data}

        # Convert the output to a JSON string
        output_str = json.dumps(output, indent=4)

        # Display the JSON-like format
        st.write("### Data in JSON-like Format:")
        st.json(output)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a `.npy` file to view its contents.")
