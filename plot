
from datetime import datetime
import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

# Paths
inference_set = ['../Dataset/prints_to_test_25/']
CoarseNet_path = '../Models/CoarseNet.h5'
output_dir = f'../output_CoarseNet_25/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(output_dir, exist_ok=True)

# Load CoarseNet model (ensure `CoarseNetmodel` is defined elsewhere)
main_net_model = CoarseNetmodel((None, None, 1), CoarseNet_path, mode='deploy')

# Process each dataset
for deploy_set in inference_set:
    set_name = deploy_set.split('/')[-2]
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)
    
    output_set_dir = os.path.join(output_dir, set_name, 'enhance_img_results')
    os.makedirs(output_set_dir, exist_ok=True)

    for i, name in enumerate(img_name):
        print(f"Processing {i+1}/{len(img_name)}: {name}")
        
        # Load image
        image_path = os.path.join(deploy_set, 'img_files', f'{name}.bmp')
        image = imread(image_path, as_gray=True)
        img_size = np.array(image.shape, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]
        
        # Prepare input
        image_input = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
        
        # Model prediction
        enh_img, enh_img_imag, enhance_img, *_ = main_net_model.predict(image_input)

        # Print shape and values
        print(f"Enh_img: shape={enh_img.shape}, min={np.min(enh_img)}, max={np.max(enh_img)}")
        print(f"Enh_img_imag: shape={enh_img_imag.shape}, min={np.min(enh_img_imag)}, max={np.max(enh_img_imag)}")
        print(f"Enhance_img: shape={enhance_img.shape}, min={np.min(enhance_img)}, max={np.max(enhance_img)}")

        # Save raw outputs directly
        np.save(os.path.join(output_set_dir, f"{name}_enh_img.npy"), enh_img)
        np.save(os.path.join(output_set_dir, f"{name}_enh_img_imag.npy"), enh_img_imag)
        np.save(os.path.join(output_set_dir, f"{name}_enhance_img.npy"), enhance_img)

        # Display images
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(enh_img[0, ..., 0], cmap='gray')
        plt.title("Enh Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(enh_img_imag[0, ..., 0], cmap='gray')
        plt.title("Enh Image Imaginary")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(enhance_img[0, ..., 0], cmap='gray')
        plt.title("Enhance Image")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save visualizations to disk
        plt.savefig(os.path.join(output_set_dir, f"{name}_visualization.jpg"))
        plt.close()
