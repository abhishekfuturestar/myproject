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

# Initialize logging
logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = False

# Helper function to create directories
def mkdir(directory):
    os.makedirs(directory, exist_ok=True)

# Helper function to save and display images
def save_and_display_images(image, output_dir, image_name):
    """
    Save and optionally display the image.
    Normalize the image before saving it.
    """
    mkdir(output_dir)
    # Normalize the image
    img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    img_normalized = img_normalized.astype(np.uint8)
    
    # Save the image
    save_path = os.path.join(output_dir, f"{image_name}.png")
    cv2.imwrite(save_path, img_normalized)
    
    # Display the image
    plt.figure()
    plt.imshow(img_normalized, cmap='gray')
    plt.title(image_name)
    plt.axis('off')
    plt.show()

for deploy_set in inference_set:
    set_name = deploy_set.split('/')[-2]
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    # Create output directories for each variable
    enh_img_dir = os.path.join(output_dir, set_name, 'enh_img')
    enh_img_imag_dir = os.path.join(output_dir, set_name, 'enh_img_imag')
    enhance_img_dir = os.path.join(output_dir, set_name, 'enhance_img')

    mkdir(enh_img_dir)
    mkdir(enh_img_imag_dir)
    mkdir(enhance_img_dir)

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
        image = imread(os.path.join(deploy_set, 'img_files', f"{name}.bmp"), as_gray=True)
        img_size = np.array(image.shape, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]

        # Prepare image for model prediction
        image_input = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        # Model prediction
        outputs = main_net_model.predict(image_input)
        print(f"Outputs received: {len(outputs)}")  # Debugging to understand outputs
        
        # Handle outputs dynamically
        enh_img = outputs[0] if len(outputs) > 0 else None
        enh_img_imag = outputs[1] if len(outputs) > 1 else None
        enhance_img = outputs[2] if len(outputs) > 2 else None

        # Save and display the images for each variable
        if enh_img is not None:
            save_and_display_images(enh_img[0, :, :, 0], enh_img_dir, f"enh_img_{name}")
        if enh_img_imag is not None:
            save_and_display_images(enh_img_imag[0, :, :, 0], enh_img_imag_dir, f"enh_img_imag_{name}")
        if enhance_img is not None:
            save_and_display_images(enhance_img[0, :, :, 0], enhance_img_dir, f"enhance_img_{name}")

print("All images have been processed and saved.")
