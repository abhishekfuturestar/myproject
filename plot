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

            # Convert the enhanced image to uint8 range [0, 255]
            enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)

            # Create a mask of the non-black regions (regions where the input is not completely black)
            mask = (image > 0).astype(np.uint8)

            # Replace any remaining black pixels in the output with white
            final_image = np.where(mask == 1, enhanced_image_uint8, 255)

            # Ensure all black pixels (value 0) are converted to white in the output
            final_image[final_image == 0] = 255

            # Generate the output file path
            output_path = os.path.join(output_folder, f"enhanced_{file_name}")

            # Save the output image
            imsave(output_path, final_image)
            print(f"Enhanced image saved to: {output_path}")

# Define the input and output folders
input_folder = 'C:/Users/2179048/Desktop/ridge_ compen/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_ compen/output'

apply_ridge_compensation_filter(input_folder, output_folder)
