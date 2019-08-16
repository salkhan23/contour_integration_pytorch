import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import gabor_fits
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


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10
    plt.ion()
    np.random.seed(random_seed)

    base_data_dir = './data/single_frag_fulltile_32_fragtile_20'

    frag_size = np.array([20, 20])
    full_tile_size = np.array([32, 32])
    image_size = np.array([512, 512, 3])

    num_train_images_per_set = 300
    num_val_images_per_set = 50

    # Gabor Fragment
    gabor_parameters = [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 90,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }]

    fragment = gabor_fits.get_gabor_fragment(gabor_parameters, frag_size)

    contour_len_arr = [3, 5, 7, 9, 12]
    beta_rotation_arr = [0, 15, 30]
    alpha_rotation_arr = [0]

    # Generate the training Set
    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_train_images_per_set,
        base_dir=os.path.join(base_data_dir, 'train'),
        frag=fragment,
        frag_params=gabor_parameters,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
    )

    # Generate the Validation Set
    fields1993_stimuli.generate_data_set(
        n_imgs_per_set=num_val_images_per_set,
        base_dir=os.path.join(base_data_dir, 'val'),
        frag=fragment,
        frag_params=gabor_parameters,
        c_len_arr=contour_len_arr,
        beta_rot_arr=beta_rotation_arr,
        alpha_rot_arr=alpha_rotation_arr,
        f_tile_size=full_tile_size,
        img_size=image_size,
    )

    # Channel wise mean and standard deviation
    list_of_files = []

    train_images = get_list_of_image_files(os.path.join(base_data_dir, 'train'))
    list_of_files.extend(train_images)

    val_images = get_list_of_image_files(os.path.join(base_data_dir, 'val'))
    list_of_files.extend(val_images)

    mean, std = get_dataset_mean_and_std(list_of_files)
    print("Dataset Channel-wise\n mean {} \n std {}".format(mean, std))

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
        'g_params': gabor_parameters
    }

    metadata_filename = 'dataset_meta_data'
    txt_file = os.path.join(base_data_dir, 'dataset_metadata' + '.txt')
    pkl_file = os.path.join(base_data_dir, 'dataset_metadata' + '.pickle')

    handle = open(txt_file, 'w+')
    for k, v in meta_data.items():
        handle.write(k + ': ' + str(v) + '\n')
    handle.close()

    with open(pkl_file, 'wb') as handle:
        pickle.dump(meta_data, handle)

    input("press any key to exit")
