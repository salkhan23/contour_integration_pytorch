# ---------------------------------------------------------------------------------------
# Scatter plot predictions of model vs control.
# This is a new metric to check how well the model is doing compared to control at
# predicting edges and non edges.
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb
import pdb
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markeredgewidth': 3

})


def get_above_below_on_diagonal_counts(x_axis, y_axis, l_th, h_th):
    """

    """

    below_h_th = x_axis < h_th
    above_l_th = x_axis >= l_th
    bin_mask = below_h_th * above_l_th

    x_axis_in_bin = x_axis * bin_mask
    y_axis_for_bin = y_axis * bin_mask

    above = np.count_nonzero(y_axis_for_bin > x_axis_in_bin)
    below = np.count_nonzero(y_axis_for_bin < x_axis_in_bin)
    total = np.count_nonzero(x_axis_in_bin)

    # print("bin [{:0.1f}, {:0.1f}]: Above {},below {}, total {}".format(
    #     l_th, h_th, above, below, total))

    return above, below, (total - (above + below)), x_axis_in_bin, y_axis_for_bin


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------

    model_predictions_dir = './results/predictions/biped_dataset_val/model'
    control_predictions_dir = './results/predictions/biped_dataset_val/control'
    gt_dir = './results/predictions/biped_dataset_val/gt'

    th_arr = np.arange(0, 1.1, 0.1)

    # Immutable ------------------------
    plt.ion()

    # All Required paths exists
    if not os.path.exists(model_predictions_dir):
        raise Exception("Model predictions directory {} DNE !".format(model_predictions_dir))
    if not os.path.exists(model_predictions_dir):
        raise Exception("Control predictions directory {} DNE !".format(control_predictions_dir))
    if not os.path.exists(gt_dir):
        raise Exception("ground truth directory {} DNE !".format(gt_dir))

    # Predictions, Labels and GT have the required number and identical list of files
    list_of_files = os.listdir(gt_dir)
    model_list_of_files = os.listdir(model_predictions_dir)
    control_list_of_files = os.listdir(control_predictions_dir)

    if len(list_of_files) != len(model_list_of_files) != len(control_list_of_files):
        raise Exception(
            "Number of files for Model {}, Control {} and Ground Truth {} do not match".format(
                model_list_of_files, control_list_of_files, list_of_files))

    # For plotting the scatter figure
    x = np.linspace(0, 1, 100)  # for plotting the diagonal
    scat_edges_fig, scat_edges_ax = plt.subplots()
    scat_non_edges_fig, scat_non_edges_ax = plt.subplots()

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    # For each threshold bin: above, below, on
    edges_count = np.zeros((len(th_arr) - 1, 3))
    non_edges_count = np.zeros_like(edges_count)

    for idx, img in enumerate(sorted(list_of_files)):
        print("[{}] processing image: {}".format(idx, img))

        gt = np.asarray(
            Image.open(os.path.join(gt_dir, img)).convert("1"))
        model_out = np.asarray(
            Image.open(os.path.join(model_predictions_dir, img)).convert("L")) / 255.
        control_out = np.asarray(
            Image.open(os.path.join(control_predictions_dir, img)).convert("L")) / 255.

        # -------------------------------------------------------------------------------
        # Process Edges
        # -------------------------------------------------------------------------------
        edges_model = (model_out * gt).flatten()
        edges_control = (control_out * gt).flatten()

        for bin_idx in range(len(th_arr) - 1):
            low_th = th_arr[bin_idx]
            high_th = th_arr[bin_idx + 1]

            above_diag, below_diag, on_diag, x_in_bin, y_in_bin = \
                get_above_below_on_diagonal_counts(edges_control, edges_model, low_th, high_th)

            edges_count[bin_idx, 0] += above_diag
            edges_count[bin_idx, 1] += below_diag
            edges_count[bin_idx, 2] += on_diag

            # # Plot
            # scat_edges_ax.scatter(x_in_bin, y_in_bin)

        # -------------------------------------------------------------------------------
        # Process Non-Edges
        # -------------------------------------------------------------------------------
        non_edges_mask = np.abs(gt - 1)
        non_edges_model = (model_out * non_edges_mask).flatten()
        non_edges_control = (control_out * non_edges_mask).flatten()

        for bin_idx in range(len(th_arr) - 1):
            low_th = th_arr[bin_idx]
            high_th = th_arr[bin_idx + 1]

            above_diag, below_diag, on_diag, x_in_bin, y_in_bin = \
                get_above_below_on_diagonal_counts(
                    non_edges_control, non_edges_model, low_th, high_th)

            non_edges_count[bin_idx, 0] += above_diag
            non_edges_count[bin_idx, 1] += below_diag
            non_edges_count[bin_idx, 2] += on_diag

            # # Plot
            # scat_non_edges_ax.scatter(x_in_bin, y_in_bin)

    # -----------------------------------------------------------------------------------
    # Plot Edges Count
    # -----------------------------------------------------------------------------------
    scat_edges_ax.plot(x, x, color='red', linewidth=3)
    scat_edges_ax.set_xlabel("Control")
    scat_edges_ax.set_ylabel("Model")
    scat_edges_ax.set_title("Edges Scatter Plot")

    print("Edges Count {}".format('*' * 80))
    for bin_idx in range(len(th_arr) - 1):
        print("bin [{:0.1f},{:0.1f}]. above {}, below {}, on {}".format(
            th_arr[bin_idx], th_arr[bin_idx + 1],
            edges_count[bin_idx, 0], edges_count[bin_idx, 1], edges_count[bin_idx, 2]))

    # Plot Above, below and On diagonal counts for each threshold bin
    plt.figure()
    plt.plot(th_arr[1:], edges_count[:, 0], label='above_diagonal', marker='x', markersize=10)
    plt.plot(th_arr[1:], edges_count[:, 1], label='below_diagonal', marker='x', markersize=10)
    plt.plot(th_arr[1:], edges_count[:, 2], label='on_diagonal', marker='x', markersize=10)
    plt.xlabel("bin")
    plt.xlim([0, 1])
    plt.ylabel("Count")
    plt.title("Edges. Total Diagonal: Above {}, Below {}, On {}".format(
        edges_count[:, 0].sum(), edges_count[:, 1].sum(), edges_count[:, 2].sum()))
    plt.grid()
    plt.legend()

    # -----------------------------------------------------------------------------------
    # Plot Non-Edges Count
    # -----------------------------------------------------------------------------------
    scat_non_edges_ax.plot(x, x, color='red', linewidth=3)
    scat_non_edges_ax.set_xlabel("Control")
    scat_non_edges_ax.set_ylabel("Model")
    scat_non_edges_ax.set_title("Non Edges Scatter Plot")

    print("Non-Edges Count {}".format('*' * 80))
    for bin_idx in range(len(th_arr) - 1):
        print("bin [{:0.1f},{:0.1f}]. above {}, below {}, on {}".format(
            th_arr[bin_idx], th_arr[bin_idx + 1],
            edges_count[bin_idx, 0], edges_count[bin_idx, 1], edges_count[bin_idx, 2]))

    # Plot Above, below and On diagonal counts for each threshold bin
    plt.figure()
    plt.plot(th_arr[1:], non_edges_count[:, 0], label='above_diagonal', marker='x', markersize=10)
    plt.plot(th_arr[1:], non_edges_count[:, 1], label='below_diagonal', marker='x', markersize=10)
    plt.plot(th_arr[1:], non_edges_count[:, 2], label='on_diagonal', marker='x', markersize=10)
    plt.xlabel("bin")
    plt.xlim([0, 1])
    plt.ylabel("Count")
    plt.title("Non-Edges. Total Diagonal: Above {}, Below {}, On {}".format(
        non_edges_count[:, 0].sum(), non_edges_count[:, 1].sum(), non_edges_count[:, 2].sum()))
    plt.grid()
    plt.legend()


# ---------------------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------------------
print("End")
pdb.set_trace()
