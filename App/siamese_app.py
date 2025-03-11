import os
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

#Configurations
IMG_SIZE = (128, 128)
REAL_DIR = "processed_dataset_npy/real" 
MODEL_PATH = "Siamese.keras"  
THRESHOLD = 0.95 

# custom functions for the Lambda layer
def abs_diff(tensors):
    return tf.abs(tensors[0] - tensors[1])

def output_shape(input_shapes):
    return input_shapes[0]

# Load the trained Siamese model with custom_objects
@st.cache_resource
def load_siamese_model(model_path):
    custom_objs = {'abs_diff': abs_diff, 'output_shape': output_shape}
    model = load_model(model_path, custom_objects=custom_objs)
    return model

siamese_model = load_siamese_model(MODEL_PATH)

# Preprocessing the Image
def preprocess_image(image):
    """
    Preprocesses an image by cleaning noise, resizing, and normalizing.
    Returns the processed image as a float32 array in [0, 1].
    """
    # Convert PIL image to NumPy array and then to BGR (OpenCV format)
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    _, binary_mask = cv2.threshold(resized, 240, 255, cv2.THRESH_BINARY)
    blank_canvas = np.ones_like(resized, dtype=np.uint8) * 255
    blank_canvas[binary_mask == 0] = resized[binary_mask == 0]
    normalized = blank_canvas.astype("float32") / 255.0
    
    # Expand dims to match (128,128,1)
    return np.expand_dims(normalized, axis=-1)

# Loading Genuine Signature for the request user
def load_genuine_signatures(user_id):
    """
    Loads all preprocessed .npy signature images for the given user.
    """
    user_dir = os.path.join(REAL_DIR, user_id)
    if not os.path.isdir(user_dir):
        return None
    
    imgs = []
    for file in os.listdir(user_dir):
        if file.endswith(".npy"):
            img = np.load(os.path.join(user_dir, file))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            imgs.append(img)
    return imgs if imgs else None

def verify_user_signature(user_id, query_img):
    """
    Verifies whether the provided signature image belongs to the claimed user.
    Returns a tuple (verified: bool, avg_score: float, scores: list).
    """
    genuine_imgs = load_genuine_signatures(user_id)
    if genuine_imgs is None:
        return None, 0.0, []
    
    scores = []
    for genuine in genuine_imgs:
        query_batch = np.expand_dims(query_img, axis=0)
        genuine_batch = np.expand_dims(genuine, axis=0)
        score = siamese_model.predict([query_batch, genuine_batch])[0][0]
        scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score >= THRESHOLD, avg_score, scores

st.title("Signature Verification System")
st.markdown(
    """
    Upload a signature image and provide the user ID to verify whether the signature belongs to the user.
    """
)

user_id = st.text_input("Enter User ID:")

uploaded_image = st.file_uploader("Upload Signature Image:", type=["png", "jpg", "jpeg"])

if uploaded_image and user_id:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Signature Image", use_column_width=True)

    # Preprocess the uploaded image
    query_image_pil = Image.open(uploaded_image)
    query_img = preprocess_image(query_image_pil)

    # Verify the signature
    verified, avg_score, scores = verify_user_signature(user_id, query_img)

    # Display the results
    if verified is None:
        st.error(f"No genuine signatures found for user '{user_id}'.")
    else:
        st.write(f"**Average Similarity Score:** {avg_score:.4f}")
        if verified:
            st.success("Signature VERIFIED for this user.")
        else:
            st.error("Signature NOT VERIFIED for this user.")
