import matplotlib.pyplot as plt
import numpy as np

import gabor_fits
import generate_stimuli


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    fragment_size = np.array([11, 11])
    full_tile_size = np.array([18, 18])

    image_size = np.array([227, 227, 3])

    # Gabor Fragment
    gabor_parameters = {
        'x0': 0,
        'y0': 0,
        'theta_deg': 90,
        'amp': 1,
        'sigma': 2.0,
        'lambda1': 6,
        'psi': 0,
        'gamma': 1
    }

    fragment = gabor_fits.get_gabor_fragment(gabor_parameters, fragment_size)

    # Immutable
    plt.ion()
    np.random.seed(random_seed)

    bg_value = generate_stimuli.get_mean_pixel_value_at_boundary(fragment)
    test_image = np.ones(image_size, dtype=np.uint8) * bg_value

    # -----------------------------------------------------------------------------------
    center_full_tile_start = (image_size[:2] // 2) - (full_tile_size // 2)

    bg_frag_starts = generate_stimuli.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=image_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_full_tile_start[0]
    )

    contour_len = 7
    beta_rotation = 15
    alpha_rotation = 0

    #  Add the Contour Path
    test_image, path_fragment_starts = generate_stimuli.add_contour_path_constant_separation(
        img=test_image,
        frag=fragment,
        frag_params=gabor_parameters,
        c_len=contour_len,
        beta=beta_rotation,
        alpha=alpha_rotation,
        d=full_tile_size[0],
        rand_inter_frag_direction_change=True,
        base_contour='sigmoid',
        center_frag_start=np.array([180,150])
    )

    # Generate the label
    labels = np.zeros(len(bg_frag_starts))
    bg_frag_centers = bg_frag_starts + full_tile_size // 2

    # Find the full tile in which contour fragments lie (mostly)
    for path_frag_start in path_fragment_starts:

        path_frag_center = path_frag_start + fragment_size // 2

        dist_to_c_frag = np.linalg.norm(bg_frag_centers - path_frag_center, axis=1)
        closest_full_tile_idx = np.argmin(dist_to_c_frag)

        print(closest_full_tile_idx, dist_to_c_frag[closest_full_tile_idx])

        labels[closest_full_tile_idx] = 1

    z = np.reshape(labels, (13, 13))
    print(z)


    img = generate_stimuli.highlight_tiles(test_image, full_tile_size, bg_frag_starts, edge_color=(0, 255, 0))

    plt.figure()
    plt.imshow(img)

    import pdb
    pdb.set_trace()

    # # -----------------------------------------------------------------------------------
    # #  Stimulus - Single function
    # # -----------------------------------------------------------------------------------
    # contour_len = 9
    # beta_rotation = 15
    # alpha_rotation = 0
    #
    # img_arr = generate_stimuli.generate_contour_images(
    #     n_images=1,
    #     frag=fragment,
    #     frag_params=gabor_parameters,
    #     c_len=contour_len,
    #     beta=beta_rotation,
    #     alpha=alpha_rotation,
    #     f_tile_size=full_tile_size,
    #     img_size=np.array((227, 227, 3)),
    #     random_alpha_rot=True
    # )
    #
    # plt.figure()
    # image_idx = 0
    # plt.imshow(img_arr[image_idx, :])
    # plt.title("Image @ index {}".format(image_idx))