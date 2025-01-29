import cv2
import numpy as np
from skimage.filters import meijering
from skimage.io import imsave
from skimage.metrics import structural_similarity as ssim
import os

def apply_ridge_compensation_filter(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List to store SSIM scores
    ssim_scores = []

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        if os.path.isfile(input_path):
            # Read the image (grayscale)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read the image from {input_path}")
                continue

            # Normalize the image to the range [0, 1]
            normalized_image = image / 255.0

            # Create a mask for the fingerprint region (non-white areas)
            mask = (normalized_image < 1).astype(np.uint8)

            # Apply the Meijering filter to the image
            enhanced_image = meijering(normalized_image, sigmas=range(1, 5), black_ridges=False)

            # Preserve the white background and only enhance the fingerprint
            combined_image = np.where(mask == 1, enhanced_image, 1.0)  # Keep the background white

            # Convert the result back to [0, 255] range
            final_image = (combined_image * 255).astype(np.uint8)

            # Apply the bitwise NOT operation on the enhanced (final) image
            bitwise_image = cv2.bitwise_not(final_image)

            # Calculate SSIM between the original image and the bitwise-not of the enhanced image
            score, _ = ssim(image, bitwise_image, full=True, data_range=255)

            # Save the SSIM score for later use
            ssim_scores.append((file_name, score))

            # Add SSIM score as text on the combined image
            font = cv2.FONT_HERSHEY_SIMPLEX
            score_text = f"SSIM: {score:.4f}"

            color = (0, 0, 255)  # Red color for SSIM score text

            # Convert grayscale images to BGR before adding colored text
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            bitwise_bgr = cv2.cvtColor(bitwise_image, cv2.COLOR_GRAY2BGR)

            # Concatenate original image and the bitwise-enhanced image side by side
            combined_image = np.hstack((image_bgr, bitwise_bgr))

            # Place SSIM text on the combined image (at the top of the image)
            combined_image_with_text = cv2.putText(
                combined_image, score_text, (10, 30), font, 0.8, color, 2, cv2.LINE_AA
            )

            # Save the combined image with SSIM score to the output folder
            output_path = os.path.join(output_folder, f"combined_{file_name}")
            imsave(output_path, combined_image_with_text)

            print(f"Saved combined image: {output_path} with SSIM: {score:.4f}")

    # Print SSIM scores for all images
    print("\nSSIM Scores for all images:")
    for file_name, score in ssim_scores:
        print(f"{file_name}: SSIM = {score:.4f}")

    print("Processing complete. All combined images saved.")


# Define input and output folders
input_folder = 'C:/Users/2179048/Desktop/Ridge_Comp_Mejering/input'
output_folder = 'C:/Users/2179048/Desktop/Ridge_Comp_Mejering/final'

apply_ridge_compensation_filter(input_folder, output_folder)
