import matplotlib.pyplot as plt
import numpy as np

import gabor_fits
import fields1993_stimuli


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10
    plt.ion()
    np.random.seed(random_seed)

    frag_size = np.array([11, 11])
    full_tile_size = np.array([18, 18])
    image_size = np.array([227, 227, 3])

    # Gabor Fragment
    gabor_parameters = [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 90,
        'amp': 1,
        'sigma': 2.0,
        'lambda1': 6,
        'psi': 0,
        'gamma': 1
    }]

    fragment = gabor_fits.get_gabor_fragment(gabor_parameters, frag_size)

    contour_len_arr = [3, 5, 7, 9]
    beta_rotation_arr = [0, 15, 30]
    alpha_rotation_arr = [0]

    # ---------------------------------------------
    # Generate the training Set
    data_dir = './data/curved_contours/train'
    num_images_per_set = 300

    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_images_per_set,
        base_dir=data_dir,
        frag=fragment,
        frag_params=gabor_parameters,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
    )

    # Generate the Test Set
    data_dir = './data/curved_contours/test'
    num_images_per_set = 50

    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_images_per_set,
        base_dir=data_dir,
        frag=fragment,
        frag_params=gabor_parameters,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
    )




    #
    # # -----------------------------------------------------------------------------------
    # # Generate multiple images and store data and labels in directory
    # # -----------------------------------------------------------------------------------
    # if os.path.exists(base_dir):
    #     ans = input("Data already exists in {}.\nEnter y to overwrite".format(base_dir))
    #     if 'y' in ans.lower():
    #         shutil.rmtree(base_dir)
    #     else:
    #         sys.exit()
    #
    # data_key = 'data_key.txt'  # Stores the path to all files in the image_set
    #
    # data_store_dir = os.path.join(base_dir, 'images')
    # labels_store_dir = os.path.join(base_dir, 'labels')
    # data_key_file = os.path.join(base_dir, data_key)
    #
    # start_time = datetime.now()
    # n_total_images = 0
    #
    # if not os.path.exists(data_store_dir):
    #     os.makedirs(data_store_dir)
    # if not os.path.exists(labels_store_dir):
    #     os.makedirs(labels_store_dir)
    #
    # f = open(data_key_file, 'w+')
    #
    # for c_len in c_len_arr:
    #     c_len_name = 'clen_{}'.format(c_len)
    #
    #     x_start_range, y_start_range = fields1993_stimuli.get_contour_start_ranges(
    #         c_len=c_len,
    #         frag_orient=gabor_params[0]['theta_deg'],  # Todo handle the case when there are three orientations
    #         f_tile_size=f_tile_size,
    #         img_size=image_size
    #     )
    #
    #     for beta in beta_rot_arr:
    #         c_len_beta_rot_dir = os.path.join(c_len_name, 'beta_{}'.format(beta))
    #
    #         for alpha in alpha_rot_arr:
    #             c_len_beta_rot_alpha_rot_dir = os.path.join(c_len_beta_rot_dir, 'alpha_{}'.format(alpha))
    #
    #             store_data_dir_full = os.path.join(data_store_dir, c_len_beta_rot_alpha_rot_dir)
    #             store_label_dir_full = os.path.join(labels_store_dir, c_len_beta_rot_alpha_rot_dir)
    #
    #             if not os.path.exists(store_data_dir_full):
    #                 os.makedirs(store_data_dir_full)
    #             if not os.path.exists(store_label_dir_full):
    #                 os.makedirs(store_label_dir_full)
    #
    #             print("Generating {} images with c_len = {}, beta = {}, alpha = {}".format(
    #                 n_imgs_per_set, c_len, beta, alpha))
    #
    #             for i_idx in range(n_imgs_per_set):
    #
    #                 center_frag_start = np.array([
    #                     np.random.randint(x_start_range[0], x_start_range[1]),
    #                     np.random.randint(y_start_range[0], y_start_range[1]),
    #                 ])
    #
    #                 img, img_label = fields1993_stimuli.generate_contour_image(
    #                     frag=frag,
    #                     frag_params=gabor_params,
    #                     c_len=c_len,
    #                     beta=beta,
    #                     alpha=alpha,
    #                     f_tile_size=f_tile_size,
    #                     img_size=image_size,
    #                     random_alpha_rot=False,
    #                     center_frag_start=center_frag_start
    #                 )
    #
    #                 # Save
    #                 file_name = 'clen_{}_beta_{}_alpha_{}_{}'.format(c_len, beta, alpha, i_idx)
    #                 plt.imsave(fname=os.path.join(store_data_dir_full, file_name + '.png'), arr=img, format='PNG')
    #                 np.save(file=os.path.join(store_label_dir_full, file_name + '.npy'), arr=img_label)
    #                 f.write(os.path.join(c_len_beta_rot_alpha_rot_dir, file_name) + '\n')
    #
    #                 n_total_images += 1
    # f.close()

    # -----------------------------------------------------------------------------------

    input("press any key to exit")
