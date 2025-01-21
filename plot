import cv2
import numpy as np
from skimage.filters import meijering
from skimage.io import imsave
import os

def apply_ridge_compensation_filter(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        if os.path.isfile(input_path):
            # Read the image (grayscale)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read the image from {input_path}")
                continue

            # Normalize the image to range [0, 1]
            normalized_image = image / 255.0

            # Apply the Meijering filter to enhance ridges
            enhanced_image = meijering(normalized_image, sigmas=range(1, 5), black_ridges=False)

            # Create a mask for the fingerprint region (non-white areas)
            mask = (image < 255).astype(np.uint8)  # Mask non-white (fingerprint) areas

            # Preserve fingerprint areas and set everything else (black) to white
            final_image = np.where(mask == 1, (enhanced_image * 255).astype(np.uint8), 255)

            # Save the output image
            output_path = os.path.join(output_folder, f"enhanced_{file_name}")
            imsave(output_path, final_image)
            print(f"Enhanced image saved to: {output_path}")

# Define the input and output folders
input_folder = 'C:/Users/2179048/Desktop/ridge_compen2/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_compen2/output10'

apply_ridge_compensation_filter(input_folder, output_folder)
