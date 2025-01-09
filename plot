ValueError                                Traceback (most recent call last)
Cell In[3], line 89
     86 image_input = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
     88 # Model prediction
---> 89 enh_img, enh_img_imag, enhance_img = main_net_model.predict(image_input)
     91 # Save and display the images for each variable
     92 save_and_display_images(enh_img[0, :, :, 0], enh_img_dir, f"enh_img_{name}")

ValueError: too many values to unpack (expected 3)
