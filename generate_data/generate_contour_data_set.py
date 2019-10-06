# ----------------------------------------------------------------------------
#  Generate Contour Data set
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import fields1993_stimuli


def get_list_of_image_files(base_dir):
    data_key = os.path.join(base_dir, 'data_key.txt')
    image_dir = os.path.join(base_dir, 'images')

    with open(data_key, 'rb') as fh:
        files = [os.path.join(image_dir, line.strip().decode("utf-8") + '.png') for line in fh.readlines()]

    return files


def get_dataset_mean_and_std(files):
    """
    Compute the data set channel-wise mean and standard deviation
        Var[x] = E[X ^ 2] - E ^ 2[X]
        Ref: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9

    :param files:
    :return:
    """
    cnt = 0
    first_moment = np.zeros(3)
    second_moment = np.zeros(3)

    for file in files:
        img = plt.imread(file)  # puts data in range [0, 1], but also includes a 4th channel
        img = img[:, :, :3]

        h, w, c = img.shape
        nb_pixels = h * w

        sum_ = np.sum(img, axis=(0, 1))
        sum_of_square = np.sum(img ** 2, axis=(0, 1))

        first_moment = (cnt * first_moment + sum_) / (cnt + nb_pixels)
        second_moment = (cnt * second_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return first_moment,  np.sqrt(second_moment - first_moment ** 2)


def get_filtered_gabor_sets(file_names, gabor_set_arr):
    use_set = []

    if gabor_set_arr:
        for set_idx in gabor_set_arr:
            use_set.extend([x for x in file_names if 'frag_{}/'.format(set_idx) in x])
    else:
        use_set = file_names

    return use_set


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10
    plt.ion()
    np.random.seed(random_seed)

    num_train_images_per_set = 300
    num_val_images_per_set = 50

    frag_size = np.array([7, 7])
    full_tile_size = np.array([14, 14])
    image_size = np.array([256, 256, 3])
    image_center = image_size[0:2] // 2

    gabor_params_file = './generate_data/fitted_10_gabors_params.pickle'

    # Centrally Located contours (Li 2006 Stimuli)
    base_data_dir = './data/fitted_gabors_10_full14_frag7_centered_test'
    center_frag_start = image_center - (frag_size // 2)

    # # Randomly Located Contours
    # base_data_dir = './data/fitted_gabors_10_full14_frag7_test'
    # center_frag_start = None

    # -----------------------------------------------------------------------------------
    # gabor_parameters_list - list of list of dictionaries one for each channel
    with open(gabor_params_file, 'rb') as handle:
        gabor_parameters_list = pickle.load(handle)

    contour_len_arr = [1, 3, 5, 7, 9, 12]
    beta_rotation_arr = [0, 15]
    alpha_rotation_arr = [0]

    # num_train_images_per_set = 10
    # num_val_images_per_set = 5
    # contour_len_arr = [1, 3]
    # beta_rotation_arr = [0]
    # alpha_rotation_arr = [0]

    # Generate the training Set
    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_train_images_per_set,
        base_dir=os.path.join(base_data_dir, 'train'),
        frag_tile_size=frag_size,
        frag_params_list=gabor_parameters_list,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
        center_frag_start=center_frag_start
    )

    # Generate the Validation Set
    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_val_images_per_set,
        base_dir=os.path.join(base_data_dir, 'val'),
        frag_tile_size=frag_size,
        frag_params_list=gabor_parameters_list,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
        center_frag_start=center_frag_start
    )

    # Channel wise mean and standard deviation
    list_of_files = []

    train_images = get_list_of_image_files(os.path.join(base_data_dir, 'train'))
    list_of_files.extend(train_images)

    val_images = get_list_of_image_files(os.path.join(base_data_dir, 'val'))
    list_of_files.extend(val_images)

    print("Calculating DataSet Statistics ...")
    mean, std = get_dataset_mean_and_std(list_of_files)
    print("Overall channel-wise\n mean {} \n std {}".format(mean, std))

    # Get Mean and standard deviation of each Gabor set individually
    # --------------------------------------------------------------
    gabor_set_specific_mean_list = []
    gabor_set_specific_std_list = []
    for frag_param_idx, frag_params in enumerate(gabor_parameters_list):
        gabor_set_files = get_filtered_gabor_sets(list_of_files, [frag_param_idx])

        set_mean, set_std = get_dataset_mean_and_std(gabor_set_files)
        print("Gabor Set {}. mean {}, std {}".format(frag_param_idx, set_mean, set_std))

        gabor_set_specific_mean_list.append(set_mean)
        gabor_set_specific_std_list.append(set_std)

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
        'n_train_images_per_set': num_train_images_per_set,
        'n_val_images': len(val_images),
        'n_val_images_per_set': num_val_images_per_set,
        'g_params_list': gabor_parameters_list,
        'set_specific_means': gabor_set_specific_mean_list,
        'set_specific_std': gabor_set_specific_std_list,
    }

    metadata_filename = 'dataset_meta_data'
    txt_file = os.path.join(base_data_dir, 'dataset_metadata' + '.txt')
    pkl_file = os.path.join(base_data_dir, 'dataset_metadata' + '.pickle')

    handle = open(txt_file, 'w+')
    for k, v in sorted(meta_data.items()):
        if k is 'g_params_list':
            for g_param_idx, g_param in enumerate(v):
                handle.write('g_params_{}'.format(g_param_idx) + ': ' + str(g_param) + '\n')
        else:
            handle.write(k + ': ' + str(v) + '\n')

    handle.close()

    with open(pkl_file, 'wb') as handle:
        pickle.dump(meta_data, handle)

    input("press any key to exit")
