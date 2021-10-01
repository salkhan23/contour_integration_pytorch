# ---------------------------------------------------------------------------------------
# Scatter plot predictions of model (y-axis) vs control (x_axis) for single contours
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
# See script validate_biped_dataset to get them
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib as mpl
from matplotlib.pyplot import cm
import scatter_plot_model_vs_control_edge_v2 as scatter_plot_model_vs_control_edge

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markeredgewidth': 3

})


def main(m_pred_dir, c_pred_dir, gt_dir, win_len=0.2, bin_size=0.1):
    """

    :param m_pred_dir:
    :param c_pred_dir:
    :param gt_dir:
    :param win_len:
    :param bin_size:
    :return:
    """
    list_of_sub_dirs = os.listdir(gt_dir)
    list_of_sub_dirs.sort(key=lambda x1: float(x1.split('_')[1]))

    # For each subdirectory for each threshold counts of edges above, below & on diagonal
    list_of_edge_counts = []

    # For plotting
    color = iter(cm.rainbow(np.linspace(0, 1, len(list_of_sub_dirs))))
    fig, ax = plt.subplots(figsize=(7, 7))

    for sb_dir_idx, sb_dir in enumerate(list_of_sub_dirs):
        print("[{}] Processing contours of length {}. ".format(sb_dir_idx, sb_dir))

        sb_dir_m = os.path.join(m_pred_dir, sb_dir)
        sb_dir_c = os.path.join(c_pred_dir, sb_dir)
        sb_dir_gt = os.path.join(gt_dir, sb_dir)

        e_str_arr, m_e_diff_mean, m_e_diff_std, m_e_counts, m_ne_diff_mean, m_ne_diff_std, m_ne_counts = \
            scatter_plot_model_vs_control_edge.main(
                sb_dir_m,
                sb_dir_c,
                sb_dir_gt,
                win_len=win_len,
                bin_size=bin_size,
                process_non_edges=False,
                verbose=False
            )

        c = next(color)

        scatter_plot_model_vs_control_edge.plot_prediction_differences(
                sb_dir_idx, m_e_diff_mean, m_e_diff_std, axis=ax, color=c, label='c_len_{}'.format(sb_dir))


if __name__ == "__main__":

    window_len = 1
    edge_strength_bin_size = 1

    # Note: Predictions over the validation set have to be calculated beforehand.
    # Please see validate_biped_dataset.get_predictions()
    # This is also called in the training script
    control_predictions_dir = './results/biped_new/control/random_seed_1/predictions_single_contour_natural_images_4'
    ground_truth_dir = './data/single_contour_natural_images_4/labels'

    model_predictions_dir = './results/biped_new/model/random_seed_3/predictions_single_contour_natural_images_4'
    model_color = 'b'
    model_marker = 'x'

    rpcm_predictions_dir = './results/biped_new/rpcm_variant/random_seed_96/predictions_single_contour_natural_images_4'
    rpcm_color = 'g'
    rpcm_marker = 'x'

    # Immutable ------------------------
    plt.ion()

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    # edge_str_arr, model_edges_diff_mean, model_edges_diff_std, _, \
    #     model_non_edges_diff_mean, model_non_edges_diff_std, _ = \
    main(
        m_pred_dir=model_predictions_dir,
        c_pred_dir=control_predictions_dir,
        gt_dir=ground_truth_dir,
        win_len=window_len, bin_size=edge_strength_bin_size)



    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
