import os
import cv2
import numpy as np
from glob import glob

# Configuration
INPUT_DIR = "dataset"  
OUTPUT_DIR = "processed_dataset_npy" 
IMG_SIZE = (128, 128)  

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {image_path}")
    
    # Convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to target size
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize the image to [0, 1]
    normalized = resized.astype("float32") / 255.0
    
    # Enhance contrast using histogram equalization
    equalized = cv2.equalizeHist((normalized * 255).astype("uint8"))
    processed = equalized.astype("float32") / 255.0
    
    return processed

def process_dataset(input_dir, output_dir):
    create_dir(output_dir)
    
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        output_class_dir = os.path.join(output_dir, class_dir)
        create_dir(output_class_dir)
        
        image_files = glob(os.path.join(class_path, "*.*"))
        for img_file in image_files:
            try:
                processed_img = preprocess_image(img_file)
                # Save the processed image as a .npy file
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                save_path = os.path.join(output_class_dir, base_name + ".npy")
                np.save(save_path, processed_img)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"Preprocessing complete. Processed data are saved in '{output_dir}'.")

if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)
