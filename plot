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
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read the image from {input_path}")
                continue

            normalized_image = image / 255.0

            # Add padding
            padded_image = cv2.copyMakeBorder(
                normalized_image, 10, 10, 10, 10, cv2.BORDER_REFLECT
            )

            # Apply Meijering filter
            enhanced_image = meijering(padded_image, sigmas=range(1, 5), black_ridges=False)

            # Crop padding
            height, width = normalized_image.shape
            cropped_image = enhanced_image[10:10 + height, 10:10 + width]

            # Mask black regions
            masked_image = np.where(cropped_image == 0, normalized_image, cropped_image)

            # Convert to uint8
            final_image = (masked_image * 255).astype(np.uint8)

            output_path = os.path.join(output_folder, f"enhanced_{file_name}")
            imsave(output_path, final_image)
            print(f"Enhanced image saved to: {output_path}")

input_folder = 'C:/Users/2179048/Desktop/ridge_ compen/input'
output_folder = 'C:/Users/2179048/Desktop/ridge_ compen/output'

apply_ridge_compensation_filter(input_folder, output_folder)
