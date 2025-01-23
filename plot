import cv2
import os

# Define input and output folder paths
input_folder = r'C:\Users\2179048\Desktop\ridge_compen2\output'
output_folder = r'C:\Users\2179048\Desktop\ridge_compen2\output_result'

# Make sure the output folder exists, create it if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Loop through each image file
for image_file in image_files:
    # Full path to the image
    image_path = os.path.join(input_folder, image_file)
    
    # Read the image
    img = cv2.imread(image_path)
    
    if img is not None:
        # Apply bitwise NOT operation to invert colors
        inverted_img = cv2.bitwise_not(img)
        
        # Save the result to the output folder with the same filename
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, inverted_img)
        print(f"Processed and saved: {image_file}")
    else:
        print(f"Failed to read image: {image_file}")
