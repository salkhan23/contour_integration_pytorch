# ---------------------------------------------------------------------------------------
#  Compare Feed Forward Orientations and Axis of Elongation of Lateral Kernels
#  Comparison based on axis of elongation from from Sincich & Blasdel - 2001
# ---------------------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import torch

import utils
import models.new_piech_models as new_piech_models
import gabor_fits

mpl.rcParams.update({
    'font.size': 18, 'lines.linewidth': 3,
    'lines.markersize': 10,
    'lines.markeredgewidth': 3
})


def find_ff_kernel_orientations(weights):
    n = weights.shape[0]

    ori_arr = np.ones(n) * -1000
    ori_1std_arr = np.ones(n) * -1000

    for n_idx in range(n):
        print("Process channel {}".format(n_idx))
        kernel = weights[n_idx, ]
        kernel_ch_last = np.transpose(kernel, axes=(1, 2, 0))

        ori = gabor_fits.get_filter_orientation(
            kernel_ch_last, o_type='max', display_params=False)

        ori_arr[n_idx] = ori

    return ori_arr


def get_polar_vectors_of_weights(kernel, dist_arr, ori_arr):
    """

    :param kernel:  [in_ch, r, c]
    :param dist_arr:  [r, c] Centered
    :param ori_arr:   [r, c] Orientations (radians)
    :return:
    """

    mag_arr = kernel * dist_arr  # Note we include the strength of the connection. In ref, they do not.
    kernel_mask = np.where(mag_arr > 0, 1, 0)
    theta_arr = kernel_mask * ori_arr  # Don't use mag for orientations

    mag_arr = mag_arr.flatten()
    theta_arr = theta_arr.flatten()

    # remove all non-zero
    nonzero_idxs = np.nonzero(mag_arr)
    mag_arr = mag_arr[nonzero_idxs]
    theta_arr = theta_arr[nonzero_idxs]

    return mag_arr, theta_arr


def cart2pol(x, y):
    mag = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, (x + 1e-5))  # Force it to be above zero

    return mag, theta


def pol2cart(mag, theta):
    x = mag * np.cos(theta)
    y = mag * np.sin(theta)
    return x, y


def get_average_vector(mag_arr, theta_arr):
    """
     Get Vector Average of polar vectors of individual Patches as Described in Sincich & Blasdel - 2001

    :param mag_arr:
    :param theta_arr: [radians]
    :return:
    """

    # theta = theta + 180 (Double it and mod by np.pi)
    # Ref: Sincich & Blasdel 2001
    theta_arr = theta_arr * 2
    theta_arr = np.mod(theta_arr, 2 * np.pi)
    # theta_arr = theta_arr / 2

    # To get the average, convert to cartesian co-ordinates
    cart_x, cart_y = pol2cart(mag_arr, theta_arr)

    avg_cart_x = np.sum(cart_x)
    avg_cart_y = np.sum(cart_y)

    if avg_cart_x == 0:
        print("Detected negative  x values")
        import pdb
        pdb.set_trace()

    mag_avg, theta_avg = cart2pol(avg_cart_x, avg_cart_y)

    mag_avg = mag_avg / len(mag_arr)
    norm_mag_avg = mag_avg / np.mean(mag_arr)  # Normalized index of ellipticity

    theta_avg = theta_avg / 2.0
    if theta_avg < 0:
        theta_avg = theta_avg + np.pi

    return norm_mag_avg, theta_avg


def get_channel_wise_axis_of_elongation(kernel):
    """

    :param kernel: [n_out_ch, n_in_ch, n_r, n_c]
    :return: [n_out_ch, 2]  A (Mag, Theta_deg) vector for  each out channel
    """
    n_out_ch, n_in_ch, n_row, n_col = kernel.shape

    # Construct a Distance & Orientation matrix for each location & center it
    r_center = n_row // 2
    c_center = n_col // 2

    grid_r, grid_c = np.meshgrid(range(n_col), range(n_col))

    grid_r -= r_center  # Center it
    grid_c -= c_center

    dist_mat = np.sqrt(grid_r ** 2 + grid_c ** 2)
    ori_mat = np.arctan2(-grid_c, grid_r)

    # f, ax_arr = plt.subplots(1, 2, figsize=(14, 7))
    # ax_arr[0].imshow(ori_mat * 180 / np.pi)
    # ax_arr[0].set_title("Orientation @ position (deg)")
    # ax_arr[1].imshow(dist_mat)
    # ax_arr[1].set_title("Distance @ positions")

    elongation_mag_arr = []
    elongation_ori_arr = []

    for out_ch_idx in range(n_out_ch):

        ch_kernel = kernel[out_ch_idx, ]

        ch_mag_arr, ch_ori_arr_rads = get_polar_vectors_of_weights(ch_kernel, dist_mat, ori_mat)
        ch_mag_avg, ch_ori_avg = get_average_vector(ch_mag_arr, ch_ori_arr_rads)

        elongation_mag_arr.append(ch_mag_avg)
        elongation_ori_arr.append(ch_ori_avg)

    return np.array(elongation_mag_arr), np.array(elongation_ori_arr) * 180 / np.pi


def scatter_plot_ff_orientation_lat_axis_of_elongation(ff_ori, e_ori, e_mag, i_ori, i_mag):
    """

    :param ff_ori: [Deg]
    :param e_ori: [Deg]
    :param e_mag:
    :param i_ori: [Deg]
    :param i_mag:
    :return:
    """
    f, ax_arr = plt.subplots(figsize=(9, 9))

    ax_arr.scatter(ff_ori, e_ori, s=e_mag * 1000, marker='x', c='b', label='E')

    ax_arr.scatter(ff_ori, i_ori, s=i_mag * 1000, marker='x', c='r', label='I')

    ax_arr.set_xlim([0, 180])
    ax_arr.set_ylim([0, 180])
    ax_arr.legend()
    ax_arr.set_xlabel("FF Orientation (deg)")
    ax_arr.set_ylabel("Lateral Kernel orientation (deg)")

    diagonal = np.arange(0, 180, 5)
    ax_arr.plot(diagonal, diagonal, color='k')
    f.tight_layout()


def scatter_plot_ff_orientation_lat_axis_of_elongation_individually(ff_ori, e_ori, e_mag, i_ori, i_mag):
    """

    :param ff_ori: [Deg]
    :param e_ori: [Deg]
    :param e_mag:
    :param i_ori: [Deg]
    :param i_mag:
    :return:
    """
    f, ax_arr = plt.subplots(figsize=(9, 9))

    n_chan = ff_ori.shape[0]

    ax_arr.set_xlim([0, 180])
    ax_arr.set_ylim([0, 180])
    ax_arr.legend()
    ax_arr.set_xlabel("FF Orientation (deg)")
    ax_arr.set_ylabel("Lateral Kernel orientation (deg)")

    diagonal = np.arange(0, 180, 5)
    ax_arr.plot(diagonal, diagonal, color='k')
    f.tight_layout()

    for idx in range(n_chan):
        ax_arr.scatter(ff_ori[idx], e_ori[idx], s=e_mag[idx] * 1000, marker='x', c='b', label='E')
        ax_arr.scatter(ff_ori[idx], i_ori[idx], s=i_mag[idx] * 1000, marker='x', c='r', label='I')
        ax_arr.annotate(idx, (ff_ori[idx],  e_ori[idx]))
        ax_arr.annotate(idx, (ff_ori[idx], i_ori[idx]))
        print("{:02}: FF {:10.2f}, E: {:10.2f}, r={:0.4f}, I {:10.2f}, r={:0.4f}".format(
            idx, ff_ori[idx], e_ori[idx], e_mag[idx], i_ori[idx], i_mag[idx]))

        # import pdb
        # pdb.set_trace()


# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/multiple_runs_contour_dataset/positive_lateral_weights_with_BN_best_gain_curves/' \
        '/run_2' \
        '/best_accuracy.pth'

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))

    # -------------------------------------
    plt.ion()
    np.set_printoptions(precision=3, linewidth=120, suppress=True, threshold=np.inf)
    n_channels = net.edge_extract.weight.data.numpy().shape[0]

    # -----------------------------------------------------------------------------------
    # FF Kernels
    # -----------------------------------------------------------------------------------
    ff_kernels = net.edge_extract.weight.data.detach().cpu().numpy()
    ff_kernels_size = np.array(ff_kernels.shape[2:])

    # Find orientation
    # ----------------
    ff_ori_pickle_file = 'ff_kernels_orientations.pickle'

    if os.path.exists(ff_ori_pickle_file):
        with open(ff_ori_pickle_file, 'rb') as handle:
            ff_ori_arr_deg = pickle.load(handle)
    else:
        print(">>>> Find FF Kernel orientation ....")
        ff_ori_arr_deg = find_ff_kernel_orientations(ff_kernels)

    # For Gabor Fits, the vertical is at 0 & moves anticlockwise, we want orientations
    # from the positive x-axis
    aligned_ff_ori_arr_deg = ff_ori_arr_deg + 90

    # # Print the Orientations
    # print("FF kernels Gabor Fit Orientations: ")
    # for ch_idx in range(n_channels):
    #     print("ch {}: {:0.2f}".format(ch_idx, aligned_ff_ori_arr_deg[ch_idx]))
    # print("Gabor Fits for {}/{} Kernels found".format(np.count_nonzero(
    #     ~np.isnan(aligned_ff_ori_arr_deg)), n_channels))

    # Visualize FF Kernels
    # --------------------
    ff_kernel_fig, ff_kernels_ax_arr = utils.view_ff_kernels(ff_kernels)
    # Add Orientation fits to kernels
    for ch_idx in range(n_channels):

        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8

        ff_kernels_ax_arr[r_idx][c_idx].axline(
            ff_kernels_size // 2,  # pass through center of kernel
            slope=-np.tan(aligned_ff_ori_arr_deg[ch_idx] * np.pi / 180),
            c='r', linewidth=3
        )

    # -----------------------------------------------------------------------------------
    # Axis of Elongation of Lateral Kernels. - Sincich and Blasdel - 2001
    # -----------------------------------------------------------------------------------
    lateral_e_w = net.contour_integration_layer.lateral_e.weight.data.detach().cpu().numpy()
    lateral_i_w = net.contour_integration_layer.lateral_i.weight.data.detach().cpu().numpy()
    lateral_kernels_size = np.array(lateral_e_w.shape[2:])

    e_elong_mag, e_elong_ori_deg = get_channel_wise_axis_of_elongation(lateral_e_w)
    i_elong_mag, i_elong_ori_deg = get_channel_wise_axis_of_elongation(lateral_i_w)

    for ch_idx in range(n_channels):
        print("{:02}: FF {:10.2f}, E {:10.2f}, I {:10.2f}".format(
            ch_idx,
            aligned_ff_ori_arr_deg[ch_idx],
            e_elong_ori_deg[ch_idx],
            i_elong_ori_deg[ch_idx]))

    # Visualize the Lateral Kernels
    # -----------------------------
    channel_aggregator_fcn = np.sum  # For visualizing multi channel kernels
    lat_e_fig, lat_e_ax_arr, lat_i_fig, lat_i_ax_arr = utils.view_spatial_lateral_kernels(
        lateral_e_w, lateral_i_w, spatial_func=channel_aggregator_fcn)

    # # Add Axis of Elongation to the kernels
    # for ch_idx in range(n_channels):
    #     r_idx = ch_idx // 8
    #     c_idx = ch_idx - r_idx * 8
    #
    #     lat_e_ax_arr[r_idx][c_idx].axline(
    #         (7, 7),
    #         slope=-np.tan(e_elong_ori_deg[ch_idx] * np.pi / 180), c='r',
    #         linewidth=1
    #     )
    #
    #     lat_i_ax_arr[r_idx][c_idx].axline(
    #         (7, 7),
    #         slope=-np.tan(i_elong_ori_deg[ch_idx] * np.pi / 180), c='r',
    #         linewidth=1
    #     )
    # # Note: the minus is because of how the origin is defined by imshow (0,0) is at the
    # # top left corner, y increases in the opposite direction

    # -----------------------------------------------------------------------------------
    # PLot Orientation Differences
    # -----------------------------------------------------------------------------------
    scatter_plot_ff_orientation_lat_axis_of_elongation_individually(
        aligned_ff_ori_arr_deg,
        e_elong_ori_deg, e_elong_mag,
        i_elong_ori_deg, i_elong_mag,
    )

    # # -----------------------------------------------------------------------------------
    # # Normalized index of ellipticity, rn
    # # -----------------------------------------------------------------------------------
    # plt.figure(figsize=(9, 9))
    # plt.plot(e_elong_mag, label='E')
    # plt.plot(i_elong_mag, label='I')
    # plt.title("Normalized index of ellipticity, r_n")
    # plt.xlabel("Channel")
    # plt.legend()
    # plt.grid()

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
