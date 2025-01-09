import os
import cv2
import numpy as np
from datetime import datetime

# Set input and output paths
inference_set = ['../Dataset/NeetPGlow_all_bmps/']
CoarseNet_path = '../Models/CoarseNet.h5'
FineNet_path = '../Models/FineNet.h5'
output_dir = '../Output_2.8k/' + datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs(output_dir, exist_ok=True)

# Load model (assume `main_net_model` is loaded and configured correctly)
# Example: main_net_model = load_model(CoarseNet_path)

# Simulate input image (replace with actual image loading logic)
# Assuming `image` is a NumPy array with shape (batch_size, height, width, channels)
# For example: image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)
image = np.random.rand(1, 400, 400, 1)  # Replace with actual image data

# Get predictions from the model
enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(image)

# Create subdirectories for saving outputs
enh_img_dir = os.path.join(output_dir, 'enh_img')
enh_img_imag_dir = os.path.join(output_dir, 'enh_img_imag')
enhance_img_dir = os.path.join(output_dir, 'enhance_img')

os.makedirs(enh_img_dir, exist_ok=True)
os.makedirs(enh_img_imag_dir, exist_ok=True)
os.makedirs(enhance_img_dir, exist_ok=True)

# Function to save images
def save_images(output_array, output_dir, prefix="img"):
    if len(output_array.shape) == 3:  # If single channel images
        output_array = np.expand_dims(output_array, axis=-1)
    for idx, img in enumerate(output_array):
        # Normalize and convert to uint8 for saving
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
        img = img.astype(np.uint8)
        save_path = os.path.join(output_dir, f"{prefix}_{idx}.png")
        cv2.imwrite(save_path, img)

# Save each variable to its corresponding folder
save_images(enh_img, enh_img_dir, prefix="enh_img")
save_images(enh_img_imag, enh_img_imag_dir, prefix="enh_img_imag")
save_images(enhance_img, enhance_img_dir, prefix="enhance_img")

print("Images have been saved to their respective directories.")
