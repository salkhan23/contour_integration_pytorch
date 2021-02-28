# ---------------------------------------------------------------------------------------
# PLot  FF and lateral kernels.
# Do gabor fits on them
# And look at Gabor fit orientation Differences
# ---------------------------------------------------------------------------------------

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import torch

import gabor_fits
import models.new_piech_models as new_piech_models
import utils


def find_ff_kernel_orientations(weights):
    n = weights.shape[0]
    ori_arr = np.ones(n) * -1000

    for n_idx in range(n):
        kernel = weights[n_idx, ]
        kernel_ch_last = np.transpose(kernel, axes=(1, 2, 0))

        ori_arr[n_idx] = gabor_fits.get_filter_orientation(kernel_ch_last, o_type='max', display_params=False)

    return ori_arr


def find_lateral_kernels_orientations(weights):
    """

    :param weights:  [out_ch, in_ch, r, c]
    :return:
    """
    n = weights.shape[0]
    ori_arr = np.ones(n) * -1000

    for n_idx in range(n):
        print("Processing kernel {}".format(n_idx))
        kernel = weights[n_idx, ]
        kernel_flattened = np.sum(kernel, axis=0)  # flattened across input channels
        kernel_flattened = np.expand_dims(kernel_flattened, axis=-1)  # channel last, expanded

        ori_arr[n_idx] = gabor_fits.get_filter_orientation(kernel_flattened, o_type='max', display_params=False)

    return ori_arr


def plot_ff_lat_orientation_differences(ff_ori, lateral_e_ori, lateral_i_ori):
    """

    :param ff_ori:
    :param lateral_e_ori:
    :param lateral_i_ori:
    :return:
    """

    f, ax_arr = plt.subplots(figsize=(9, 9))

    ax_arr.scatter(ff_ori, lateral_e_ori, label='E')
    ax_arr.scatter(ff_ori, lateral_i_ori, label='I')
    ax_arr.set_xlim([-90, 90])
    ax_arr.set_ylim([-90, 90])
    ax_arr.legend()
    ax_arr.set_xlabel("FF Orientation (deg)")
    ax_arr.set_ylabel("Lateral Kernel orientation (deg)")

    # draw the diagonal
    diag = np.arange(-90, 91, 5)
    ax_arr.plot(diag, diag, color='k')

    plt.title("Gabor Fits Orientations")

    return f, ax_arr


def find_ff_and_lateral_orientation_preferences(model, results_store_dir=None):
    """

    :param model:
    :param results_store_dir:
    :return:
    """
    n_channels = model.edge_extract.weight.shape[0]

    # -----------------------------------------------------------------------------------
    # Plot FF and Lateral kernels
    # -----------------------------------------------------------------------------------
    utils.view_ff_kernels(model.edge_extract.weight.data.numpy(), results_store_dir=results_store_dir)
    utils.view_spatial_lateral_kernels(
        model.contour_integration_layer.lateral_e.weight.data.numpy(),
        model.contour_integration_layer.lateral_i.weight.data.numpy(),
        spatial_func=np.mean,
        results_store_dir=results_store_dir,
    )

    # -----------------------------------------------------------------------------------
    # Get FF Kernels orientations
    # -----------------------------------------------------------------------------------
    ff_ori_pickle_filename = 'ff_kernels_orientations.pickle'
    ff_ori_pickle_file = os.path.join(results_store_dir, ff_ori_pickle_filename)

    if os.path.exists(ff_ori_pickle_file):
        with open(ff_ori_pickle_file, 'rb') as handle:
            ff_orientations = pickle.load(handle)
    else:
        print("Finding Feedforward Kernel Orientations")
        ff_orientations = find_ff_kernel_orientations(model.edge_extract.weight.data.numpy())

        if results_store_dir is not None:
            with open(ff_ori_pickle_file, 'wb') as handle:
                pickle.dump(ff_orientations, handle)

    # -----------------------------------------------------------------------------------
    # Get Lateral Kernel orientations
    # -----------------------------------------------------------------------------------
    lateral_ori_pickle_filename = 'lateral_kernels_orientation.pickle'
    lateral_ori_pickle_file = os.path.join(results_store_dir, lateral_ori_pickle_filename)

    if os.path.exists(lateral_ori_pickle_file):
        with open(lateral_ori_pickle_file, 'rb') as handle:
            lateral_kernels_orientations = pickle.load(handle)
    else:
        print("Finding Lateral Kernel Orientations")
        lateral_kernels_orientations = {
            'E': find_lateral_kernels_orientations(model.contour_integration_layer.lateral_e.weight.data.numpy()),
            'I': find_lateral_kernels_orientations(model.contour_integration_layer.lateral_i.weight.data.numpy())
        }

        if results_store_dir is not None:
            with open(lateral_ori_pickle_file, 'wb') as handle:
                pickle.dump(lateral_kernels_orientations, handle)

    print("Preferred orientations")
    for ch_idx in range(n_channels):
        print("{}: FF {} Lateral E {}, Lateral I {}".format(
            ch_idx, ff_orientations[ch_idx], lateral_kernels_orientations['E'][ch_idx],
            lateral_kernels_orientations['I'][ch_idx]))

    # -----------------------------------------------------------------------------------
    # Plot Difference of Gabor Fits
    # -----------------------------------------------------------------------------------
    f, ax_arr = plot_ff_lat_orientation_differences(
        ff_orientations,
        lateral_kernels_orientations['E'],
        lateral_kernels_orientations['I'])

    if results_store_dir is not None:
        f.savefig(os.path.join(results_store_dir, "ff_lateral_orientation_preferences_diff.jpg"))
        plt.close(f)

    return ff_orientations, lateral_kernels_orientations['E'], lateral_kernels_orientations['I']


if __name__ == '__main__':

    store_results = True

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/multiple_runs_contour_dataset/positive_lateral_weights_with_BN_best_gain_curves/' \
        '/run_5' \
        '/best_accuracy.pth'

    # -------------------------------------
    plt.ion()
    np.set_printoptions(precision=3, linewidth=120, suppress=True, threshold=np.inf)
    store_results_dir = None

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))

    if store_results:
        store_results_dir = os.path.dirname(saved_model)
        store_results_dir = os.path.join(store_results_dir, 'lateral_kernels/preferred_orientations')

        if not os.path.exists(store_results_dir):
            os.makedirs(store_results_dir)

    find_ff_and_lateral_orientation_preferences(net, results_store_dir=store_results_dir)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")

    import pdb
    pdb.set_trace()
