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

            # Apply a threshold to ensure we correctly identify the fingerprint (non-background)
            _, thresholded_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

            # Normalize the image to range [0, 1]
            normalized_image = image / 255.0

            # Apply the Meijering filter to enhance ridges
            enhanced_image = meijering(normalized_image, sigmas=range(1, 5), black_ridges=False)

            # Convert the enhanced image back to uint8
            enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)

            # Set non-fingerprint regions (background) to white in the final image
            final_image = np.where(thresholded_image == 255, 255, enhanced_image_uint8)

            # Save the output image
            output_path = os.path.join(output_folder, f"enhanced_{file_name}")
            imsave(output_path, final_image)
            print(f"Enhanced image saved to: {output_path}")

# Define the input and output folders
input_folder = 'C:/Users/2179048/Desktop/ridge_compen2/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_compen2/output10'

apply_ridge_compensation_filter(input_folder, output_folder)
