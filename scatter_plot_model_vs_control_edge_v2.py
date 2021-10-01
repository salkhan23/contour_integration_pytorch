# ---------------------------------------------------------------------------------------
# Scatter plot predictions of model (y-axis) vs control (x_axis)
# (After sigmoid but before thresholding)
# --------------------------------------------------------------------------------
#  V2: This one uses a sliding window approach and plots the relative  difference
# --------------------------------------------------------------------------------
#
# New metric to highlight differences between model & control in edge detection
# in natural images. Edge predictions are binned according to control (x_axis) edge strength
#
# The diagonal line indicates both the model and control are predicting equally.
#  Points above the diagonal indicate the model is better at detecting them
# compared to the control and visa versa.
#
# The analysis is also done for non-edges. Here the model is doing better if there
# are more points below the diagonal that above it.
#
# A summary plot for each bin is presented for all edge pints to show population trends.
#
# Requires:
#     1. Ground truth for all images
#     2. Model predictions
#     3. Control predictions
#
# See script validate_single_contour_dataset to get them
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markeredgewidth': 3

})


def get_prediction_differences(x_predicts, y_predicts, w_len, e_str_arr):
    """

    :param x_predicts:  2D Array (form multiple images flatten images)
    :param y_predicts:  2D array
    :param w_len: sliding window length
    :param e_str_arr: array of edge strengths to consider
    :return:
    """

    mu_arr = []
    sigma_arr = []
    counts_arr = []

    max_e_str = 1

    diff = y_predicts - x_predicts  # positive values mean y_predicts are better

    for idx, min_w_e_str in enumerate(e_str_arr):

        if min_w_e_str == 0:
            min_w_e_str = 0.00001  # items not in mask.

        max_w_e_str = np.min((min_w_e_str + w_len, max_e_str))

        above_min_mask = x_predicts >= min_w_e_str
        below_max_mask = x_predicts < max_w_e_str
        mask_in_range = above_min_mask * below_max_mask

        diff_in_range = diff * mask_in_range
        # remove all zeros, to calculate proper means and std
        filtered_diff_in_range = diff_in_range[np.nonzero(diff_in_range)]

        if len(filtered_diff_in_range) != 0:
            mu = np.mean(filtered_diff_in_range)
            sigma = np.std(filtered_diff_in_range)
            n_points = len(filtered_diff_in_range)
        else:
            mu = 0
            sigma = 0
            n_points = 0

        print("Range [{:0.1f} - {:0.1f}] : mean {:0.2F}, std = {:0.2f}, count = {}.".format(
            min_w_e_str, max_w_e_str, mu, sigma, n_points))

        mu_arr.append(mu)
        sigma_arr.append(sigma)
        counts_arr.append(n_points)

    mu_arr = np.array(mu_arr)
    sigma_arr = np.array(sigma_arr)
    count_arr = np.array(counts_arr)

    return mu_arr, sigma_arr, count_arr


def main(m_pred_dir, c_preds_dir, gt_dir, win_len=0.2, bin_size=0.1, process_non_edges=True, verbose=True):
    """

    :param m_pred_dir:
    :param c_preds_dir:
    :param gt_dir:
    :param win_len:
    :param bin_size:
    :param process_non_edges: [Default=True]
    :return:
    """

    # All Required paths exists
    if not os.path.exists(m_pred_dir):
        raise Exception("Model predictions directory {} DNE !".format(m_pred_dir))
    if not os.path.exists(c_preds_dir):
        raise Exception("Control predictions directory {} DNE !".format(c_preds_dir))
    if not os.path.exists(gt_dir):
        raise Exception("Ground truth directory {} DNE !".format(gt_dir))

    # Predictions, Labels and GT have the required number and identical list of files
    gt_list_of_files = os.listdir(gt_dir)
    m_list_of_files = os.listdir(m_pred_dir)
    c_list_of_files = os.listdir(c_preds_dir)

    # Windowing  Definition
    edge_strength_arr = np.arange(0, 1, bin_size)

    # ----------------------------------------------------------------------------------------------------
    if len(gt_list_of_files) != len(m_list_of_files) != len(c_list_of_files):
        raise Exception(
            "Number of files for Model {}, Control {}, Ground Truth {} do not match".format(
                len(m_list_of_files), len(c_list_of_files), len(gt_list_of_files)))

    # -----------------------------------------------------------------------------------
    #  Load All Model, Control, GT Predictions
    # -----------------------------------------------------------------------------------
    print(">>>> Loading Predictions (Number of images {}) ...".format(len(gt_list_of_files)))
    gt_full = []
    m_out_full = []
    c_out_full = []

    for i_idx, img in enumerate(sorted(gt_list_of_files)):
        if verbose:
            print("[{}] processing image: {}".format(i_idx, img))

        gt = Image.open(os.path.join(gt_dir, img)).convert("L")
        gt = np.asarray(gt.resize((256, 256), Image.LANCZOS)) / 255.0

        gt[gt >= 0.1] = 1
        gt[gt < 0.1] = 0  # Same pre-process was used during training

        m_out = np.asarray(
            Image.open(os.path.join(m_pred_dir, img)).convert("L")) / 255.
        c_out = np.asarray(
            Image.open(os.path.join(c_preds_dir, img)).convert("L")) / 255.

        # # Debug - PLots
        # f, ax_arr = plt.subplots(1, 3, figsize=(15, 5))
        # ax_arr[0].imshow(gt)
        # ax_arr[0].set_title("GT")
        # ax_arr[1].imshow(m_out * gt)
        # ax_arr[1].set_title("Model")
        # ax_arr[2].imshow(c_out * gt)
        # ax_arr[2].set_title("Control")
        # plt.tight_layout()
        #
        # import pdb
        # pdb.set_trace()

        gt_full.append(gt)
        m_out_full.append(m_out)
        c_out_full.append(c_out)

    gt_full = np.array(gt_full)
    m_out_full = np.array(m_out_full)
    c_out_full = np.array(c_out_full)

    # ---------------------------------------------------------------------------
    # Process Edges
    # ---------------------------------------------------------------------------
    # edges_gt = gt_full.flatten()
    edges_m = (m_out_full * gt_full).flatten()
    edges_c = (c_out_full * gt_full).flatten()

    print("Calculating Edge prediction differences Model vs Control...")
    m_edge_diff_means, m_edge_diff_stds, m_edge_diff_counts = get_prediction_differences(
        x_predicts=edges_c,
        y_predicts=edges_m,
        w_len=win_len,
        e_str_arr=edge_strength_arr)

    # ---------------------------------------------------------------------------
    # Process Non - Edges
    # ---------------------------------------------------------------------------
    if process_non_edges:
        non_edges_mask = np.abs(gt_full - 1)
        non_edges_model = (m_out_full * non_edges_mask).flatten()
        non_edges_control = (c_out_full * non_edges_mask).flatten()

        print("Calculating Non-Edge prediction differences Model vs Control...")
        m_non_edge_diff_means, m_non_edge_diff_stds, m_non_edge_diff_counts = get_prediction_differences(
            x_predicts=non_edges_control,
            y_predicts=non_edges_model,
            w_len=win_len,
            e_str_arr=edge_strength_arr)
    else:
        m_non_edge_diff_means = 0
        m_non_edge_diff_stds = 0
        m_non_edge_diff_counts = 0

    return edge_strength_arr, \
        m_edge_diff_means, m_edge_diff_stds, m_edge_diff_counts, \
        m_non_edge_diff_means, m_non_edge_diff_stds, m_non_edge_diff_counts


def plot_prediction_differences(
        str_bins, diff_m, diff_std, label=None, axis=None, color='black', marker='x'):
    """

    :param str_bins:
    :param diff_m:
    :param diff_std:
    :param label:
    :param axis:
    :param color:
    :param marker:
    :return:
    """

    if axis is None:
        f, axis = plt.subplots()

    axis.plot(str_bins, diff_m, marker=marker, color=color, label=label)
    axis.fill_between(
        str_bins,
        diff_m - diff_std, diff_m + diff_std,
        alpha=0.2, color=color)

    axis.axhline(y=0, color='k')
    axis.set_xlabel("Edge Strength")
    axis.set_ylabel("Prediction Difference")
    # plt.grid('ON')
    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    window_len = 0.2
    edge_strength_bin_size = 0.1

    # Note: Predictions over the validation set have to be calculated beforehand.
    # Please see validate_biped_dataset.get_predictions()
    # This is also called in the training script
    control_predictions_dir = './results/biped_new/control/random_seed_1/predictions'
    ground_truth_dir = './data/BIPED/edges/edge_maps/test/rgbr'

    model_predictions_dir = './results/biped_new/model/random_seed_96/predictions'
    model_color = 'b'
    model_marker = 'x'

    rpcm_predictions_dir = './results/biped_new/rpcm_variant/random_seed_96/predictions'
    rpcm_color = 'g'
    rpcm_marker = 'x'

    # # Old Model From Scientific Reports Paper
    # model_predictions_dir = \
    #     './results/biped' \
    #     '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base/predictions'
    #
    # control_predictions_dir = \
    #     './results/biped' \
    #     '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base/predictions'

    # Immutable - -----------------------
    plt.ion()

    edge_str_arr, model_edges_diff_mean, model_edges_diff_std, _, \
        model_non_edges_diff_mean, model_non_edges_diff_std, _ = main(
            m_pred_dir=model_predictions_dir,
            c_preds_dir=control_predictions_dir,
            gt_dir=ground_truth_dir,
            win_len=window_len, bin_size=edge_strength_bin_size, verbose=False)

    _, rpcm_edges_diff_mean, rpcm_edges_diff_std, _, \
        rpcm_non_edges_diff_mean, rpcm_non_edges_diff_std, _ = main(
            m_pred_dir=rpcm_predictions_dir,
            c_preds_dir=control_predictions_dir,
            gt_dir=ground_truth_dir,
            win_len=window_len, bin_size=edge_strength_bin_size, verbose=False)

    # Plot Edge Differences ----------------------------------------------------------------------
    _, ax = plt.subplots(figsize=(9, 9))

    plot_prediction_differences(
        edge_str_arr, model_edges_diff_mean, model_edges_diff_std,
        axis=ax, color=model_color, marker=model_marker, label='model')

    plot_prediction_differences(
        edge_str_arr, rpcm_edges_diff_mean, rpcm_edges_diff_std,
        axis=ax, color=rpcm_color, marker=rpcm_marker, label='rpcm')

    plt.title("Edges")

    # Plot Non-Edge Differences ----------------------------------------------------------------------
    _, ax = plt.subplots(figsize=(9, 9))

    plot_prediction_differences(
        edge_str_arr, model_non_edges_diff_mean, model_non_edges_diff_std,
        axis=ax, color=model_color, marker=model_marker, label='model')

    plot_prediction_differences(
        edge_str_arr, rpcm_non_edges_diff_mean, rpcm_non_edges_diff_std,
        axis=ax, color=rpcm_color, marker=rpcm_marker, label='rpcm')

    plt.title("Non-Edges")

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
