# ---------------------------------------------------------------------------------------
# Generate Li 2006 Dataset
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import generate_data_set
import fields1993_stimuli

if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    base_data_dir = './data/fields_2006_contour_length'
    gabor_params_file = 'bw_10_gabors_params.pickle'

    frag_size = np.array([20, 20])
    full_tile_size = np.array([32, 32])
    image_size = np.array([512, 512, 3])

    num_images_per_set = 50

    contour_len_arr = [1, 3, 5, 7, 9]
    beta_rotation_arr = [0]
    alpha_rotation_arr = [0]

    # --------------------------------
    plt.ion()
    np.random.seed(random_seed)

    # Gabor Fragment - list of list of dictionaries. Each Entry is a list of gabor param  dictionaries.
    # One dictionary per channel
    with open(gabor_params_file, 'rb') as handle:
        gabor_parameters_list = pickle.load(handle)

    # Put all contours in the center of the image
    image_center = image_size[0:2] // 2
    center_frag_start = image_center - (frag_size // 2)

    # Generate the training Set
    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_images_per_set,
        base_dir=os.path.join(base_data_dir, 'test'),
        frag_tile_size=frag_size,
        frag_params_list=gabor_parameters_list,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
        use_d_jitter=False,
        center_frag_start=center_frag_start,
    )

    # Channel wise mean and standard deviation
    list_of_files = []

    train_images = generate_data_set.get_list_of_image_files(os.path.join(base_data_dir, 'test'))
    list_of_files.extend(train_images)

    print("Calculating DataSet Statistics")
    mean, std = generate_data_set.get_dataset_mean_and_std(list_of_files)
    print("channel-wise\n mean {} \n std {}".format(mean, std))

    # Save a meta-data file with useful parameters
    # --------------------------------------------
    print("Saving Meta Data File")

    meta_data = {
        'full_tile_size': full_tile_size,
        'frag_tile_size': frag_size,
        'image_size': image_size,
        'c_len_arr': contour_len_arr,
        'beta_arr': beta_rotation_arr,
        'alpha_arr': alpha_rotation_arr,
        'channel_mean': mean,
        'channel_std': std,
        'n_train_images': len(train_images),
        'n_train_images_per_set': num_images_per_set,
        'n_val_images': 0,
        'n_val_images_per_set': 0,
        'g_params_list': gabor_parameters_list
    }

    metadata_filename = 'dataset_meta_data'
    txt_file = os.path.join(base_data_dir, 'dataset_metadata' + '.txt')
    pkl_file = os.path.join(base_data_dir, 'dataset_metadata' + '.pickle')

    handle = open(txt_file, 'w+')
    for k, v in meta_data.items():
        if k is 'g_params_list':
            for g_param_idx, g_param in enumerate(v):
                handle.write('g_params_{}'.format(g_param_idx) + ': ' + str(g_param) + '\n')
        else:
            handle.write(k + ': ' + str(v) + '\n')
    handle.close()

    with open(pkl_file, 'wb') as handle:
        pickle.dump(meta_data, handle)

    input("press any key to exit")
