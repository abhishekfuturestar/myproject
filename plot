
# Prepare dataset for testing. 
inference_set = ['../Dataset/prints_to_test_25/',]

CoarseNet_path = '../Models/CoarseNet.h5'

output_dir = '../output_CoarseNet_25/'+datetime.now().strftime('%Y%m%d-%H%M%S')

FineNet_path = '../Models/FineNet.h5'

logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = False

for i, deploy_set in enumerate(inference_set):
    set_name = deploy_set.split('/')[-2]

    # Read image and GT
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    mkdir(output_dir + '/'+ set_name + '/')
   # mkdir(output_dir + '/' + set_name + '/mnt_results/')
   # mkdir(output_dir + '/'+ set_name + '/seg_results/')
    # mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting \"%s\":" % (set_name))


    main_net_model = CoarseNetmodel((None, None, 1), CoarseNet_path, mode='deploy')

    # ====== Load FineNet to verify
    if isHavingFineNet == True:
        model_FineNet = FineNetmodel(num_classes=2,
                             pretrained_path=FineNet_path,
                             input_shape=(224,224,3))

        model_FineNet.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0),
                      metrics=['accuracy'])

    for i in range(0, len(img_name)):
        
        logging.info("\"%s\" %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))

        image = imread(deploy_set + 'img_files/' + img_name[i] + '.bmp', mode='L')# / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]

        original_image = image.copy()

        # Generate OF
        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)
        
        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)

        enh_img_dir = os.path.join(output_dir, 'enh_img')
        enh_img_imag_dir = os.path.join(output_dir, 'enh_img_imag')
        enhance_img_dir = os.path.join(output_dir, 'enhance_img')

        os.makedirs(enh_img_dir, exist_ok=True)
        os.makedirs(enh_img_imag_dir, exist_ok=True)
        os.makedirs(enhance_img_dir, exist_ok=True)

        def save_images(output_array, output_dir, prefix="img"):
             if len(output_array.shape) == 3:  # If single channel images
                  output_array = np.expand_dims(output_array, axis=-1)
             for idx, img in enumerate(output_array):
                 img = (img - np.min(img)) / (np.max(img) - np.min(img))* 255.0
                 img = img.astype(np.uint8)
                 save_path = os.path.join(output_dir, f"{prefix}_{idx}.png")
                 cv2.imwrite(save_path, img)

        save_images(enh_img, enh_img_dir, prefix="enh_img")
        save_images(enh_img_imag, enh_img_imag_dir, prefix="enh_img_imag")
        save_images(enhance_img, enhance_img_dir, prefix="enhance_img")
        print("Images have been saved to their respective directories.")




        
