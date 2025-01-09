inference_set = ['../Dataset/NeetPGlow_all_bmps/',]

CoarseNet_path = '../Models/CoarseNet.h5'

output_dir = '../Output_2.8k/'+datetime.now().strftime('%Y%m%d-%H%M%S')

FineNet_path = '../Models/FineNet.h5'

logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = False

enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)
