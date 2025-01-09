import os
import cv2
import numpy as np
from datetime import datetime
from skimage.io import imread
import matplotlib.pyplot as plt

# Prepare dataset for testing
inference_set = ['../Dataset/prints_to_test_25/']

CoarseNet_path = '../Models/CoarseNet.h5'
output_dir = '../output_CoarseNet_25/' + datetime.now().strftime('%Y%m%d-%H%M%S')
FineNet_path = '../Models/FineNet.h5'

logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = False

# Helper function to create directories
def mkdir(directory):
    os.makedirs(directory, exist_ok=True)

# Helper function to save and display images
def save_and_display_images(output_array, output_dir, prefix="img"):
    """
    Save images to the specified directory and display them.
    Each image is normalized and saved with a unique name.
    """
    mkdir(output_dir)
    for idx, img in enumerate(output_array):
        # Normalize the image
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
        img_normalized = img_normalized.astype(np.uint8)
        
        # Save the image
        save_path = os.path.join(output_dir, f"{prefix}_{idx}.png")
        cv2.imwrite(save_path, img_normalized)
        
        # Display the image
        plt.figure()
        plt.imshow(img_normalized, cmap='gray')
        plt.title(f"{prefix}_{idx}")
        plt.axis('off')
        plt.show()

for deploy_set in inference_set:
    set_name = deploy_set.split('/')[-2]
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    # Create output directories
    mkdir(output_dir + '/' + set_name)

    logging.info(f"Predicting \"{set_name}\":")

    main_net_model = CoarseNetmodel((None, None, 1), CoarseNet_path, mode='deploy')

    if isHavingFineNet:
        model_FineNet = FineNetmodel(num_classes=2,
                                     pretrained_path=FineNet_path,
                                     input_shape=(224, 224, 3))
        model_FineNet.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=0),
                              metrics=['accuracy'])

    for i, name in enumerate(img_name):
        logging.info(f"\"{set_name}\" {i + 1} / {len(img_name)}: {name}")
        
        # Read and preprocess the image
        image = imread(deploy_set + 'img_files/' + name + '.bmp', as_gray=True)
        img_size = np.array(image.shape, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]

        # Prepare image for model prediction
        image_input = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        # Model prediction
        enh_img, enh_img_imag, enhance_img = main_net_model.predict(image_input)

        # Define output directories for the variables
        enh_img_dir = os.path.join(output_dir, set_name, 'enh_img')
        enh_img_imag_dir = os.path.join(output_dir, set_name, 'enh_img_imag')
        enhance_img_dir = os.path.join(output_dir, set_name, 'enhance_img')

        # Save and display the images
        save_and_display_images(enh_img[0], enh_img_dir, prefix="enh_img")
        save_and_display_images(enh_img_imag[0], enh_img_imag_dir, prefix="enh_img_imag")
        save_and_display_images(enhance_img[0], enhance_img_dir, prefix="enhance_img")

print("All images have been processed and saved.")
