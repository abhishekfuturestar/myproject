
import cv2
import numpy as np
from skimage.filters import meijering
from skimage.io import imsave
import os

def apply_ridge_compensation_filter(input_folder, output_folder):
    # Check if output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        # Ensure it's a file and not a directory
        if os.path.isfile(input_path):
            # Read the image (grayscale)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read the image from {input_path}")
                continue

            # Normalize the image to range [0, 1]
            normalized_image = image / 255.0

            # Apply ridge compensation filter using Meijering filter
            enhanced_image = meijering(normalized_image, sigmas=range(1, 5), black_ridges=True)

            # Convert the result back to the range [0, 255]
            enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)

            # Generate the output file path
            output_path = os.path.join(output_folder, f"enhanced_{file_name}")

            # Save the output image
            imsave(output_path, enhanced_image_uint8)
            print(f"Enhanced image saved to: {output_path}")
# Define the input and output folders
input_folder = 'C:/Users/2179048/Desktop/ridge_ compen/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_ compen/output'

apply_ridge_compensation_filter(input_folder, output_folder)
