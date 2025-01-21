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
            enhanced_image = meijering(normalized_image, sigmas=range(1, 5), black_ridges=True)

            # Threshold to create a binary-like image
            binary_image = (enhanced_image > 0.5).astype(np.uint8)  # 0.5 is the threshold value

            # Invert the binary image so ridges are black and background is white
            inverted_image = 1 - binary_image

            # Convert to uint8 for saving (0-255 range)
            final_image = (inverted_image * 255).astype(np.uint8)

            # Generate the output file path
            output_path = os.path.join(output_folder, f"enhanced_{file_name}")

            # Save the output image
            imsave(output_path, final_image)
            print(f"Enhanced image saved to: {output_path}")

# Define the input and output folders
input_folder = 'C:/Users/2179048/Desktop/ridge_ compen/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_ compen/output'

apply_ridge_compensation_filter(input_folder, output_folder)
