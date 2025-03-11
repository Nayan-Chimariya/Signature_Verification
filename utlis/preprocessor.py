import os
import cv2
import numpy as np

# Configuration
INPUT_DIR = "Dataset"  
OUTPUT_DIR = "processed_dataset_npy" 
IMG_SIZE = (128, 128)  

def create_dir(dir_path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def preprocess_image(image_path):
    """
    Preprocesses an image by cleaning noise, resizing, and normalizing.
    Returns the processed image as a float32 array in [0, 1].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {image_path}")
    
    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resizing image to target size
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Creating a binary mask for rich strokes
    _, binary_mask = cv2.threshold(resized, 240, 255, cv2.THRESH_BINARY)
    
    # A white canvas
    blank_canvas = np.ones_like(resized, dtype=np.uint8) * 255
    
    # Transferring only rich strokes to the blank canvas
    blank_canvas[binary_mask == 0] = resized[binary_mask == 0]
    
    # Normalize the resulting canvas to range [0, 1]
    normalized = blank_canvas.astype("float32") / 255.0
    
    return normalized

def process_dataset(input_dir, output_dir):
    """
    Processes all images in the dataset recursively, saving the processed images as .npy files.
    The output directory structure will mirror the input directory structure.
    """
    for root, dirs, files in os.walk(input_dir):
        # Skip folders with no image files
        if not files:
            continue
        
        # Create the corresponding output folder
        rel_path = os.path.relpath(root, input_dir)
        output_folder = os.path.join(output_dir, rel_path)
        create_dir(output_folder)
        
        for file in files:
            # Process common image formats (adjust extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    processed_img = preprocess_image(image_path)
                    base_name = os.path.splitext(file)[0]
                    save_path = os.path.join(output_folder, base_name + ".npy")
                    np.save(save_path, processed_img)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    print(f"Preprocessing complete. Processed data are saved in '{output_dir}'.")

if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)
