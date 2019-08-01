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

    # -----------------------------------------------------------------------------------
    # Generate the training Set
    # -----------------------------------------------------------------------------------
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
    # -----------------------------------------------------------------------------------
    # Generate the test Set
    # -----------------------------------------------------------------------------------
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

    input("press any key to exit")
