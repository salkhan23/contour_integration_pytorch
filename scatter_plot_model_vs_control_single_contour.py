import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markeredgewidth': 3

})


def get_above_below_on_diagonal_counts(x_axis, y_axis, mask, l_th, h_th):
    """

    """
    if h_th == 1:
        below_h_th = x_axis <= h_th
    else:
        below_h_th = x_axis < h_th
    above_l_th = x_axis >= l_th

    bin_mask = below_h_th * above_l_th * mask

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
    model_predictions_dir = \
        './results/biped' \
        '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base' \
        '/predictions_single_contour_natural_images_4'

    control_predictions_dir = \
        './results/biped' \
        '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base' \
        '/predictions_single_contour_natural_images_4'

    gt_dir = './data/single_contour_natural_images_4/labels'

    th_arr = np.arange(0, 1.1, 0.1)

    # Immutable ------------------------
    plt.ion()

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    list_of_sub_dirs = os.listdir(gt_dir)
    list_of_sub_dirs.sort(key=lambda x1: float(x1.split('_')[1]))

    # For each subdirectory for each threshold counts of edges above, below & on diagonal
    list_of_edge_counts = []

    for sb_dir_idx, sb_dir in enumerate(list_of_sub_dirs):
        print("[{}] Processing contours of length {} {}".format(sb_dir_idx, sb_dir, '*' * 40))

        sb_dir_model = os.path.join(model_predictions_dir, sb_dir)
        sb_dir_control = os.path.join(control_predictions_dir, sb_dir)
        sb_dir_gt = os.path.join(gt_dir, sb_dir)

        # -------------------------------------------------------
        # Validate sub directories exist an contain the same data
        # -------------------------------------------------------
        if not os.path.exists(sb_dir_model):
            raise Exception("Model predictions directory {} DNE !".format(
                sb_dir_model))
        if not os.path.exists(sb_dir_control):
            raise Exception("Control predictions directory {} DNE !".format(
                sb_dir_control))
        if not os.path.exists(sb_dir_gt):
            raise Exception("ground truth directory {} DNE !".format(sb_dir_gt))

        # Predictions, Labels and GT have the required number and identical list of files
        list_of_files = os.listdir(sb_dir_gt)
        list_of_files.sort(key=lambda x1: float(x1.split('_')[1]))

        model_list_of_files = os.listdir(sb_dir_model)
        control_list_of_files = os.listdir(sb_dir_control)

        if len(list_of_files) != len(model_list_of_files) != len(control_list_of_files):
            raise Exception(
                "Number of Model {}, Control {} and GT {} files do not match!".format(
                    model_list_of_files, control_list_of_files, list_of_files))

        # For each threshold bin: above, below, on diagonal counts
        edges_count = np.zeros((len(th_arr) - 1, 3))
        prev_sums = np.zeros(3)

        # # For plotting the scatter figure
        # x = np.linspace(0, 1, 100)  # for plotting the diagonal
        # scat_edges_fig, scat_edges_ax = plt.subplots()

        # ---------------------------------------------------------
        # Process Images
        # ---------------------------------------------------------
        for idx, img in enumerate(list_of_files):
            # print("[{}] processing image: {}".format(idx, img))

            gt = np.asarray(Image.open(os.path.join(sb_dir_gt, img)).convert("L"))
            gt = (gt - gt.min()) / (gt.max() - gt.min())   # todo figure out why this is needed

            model_out = np.asarray(
                Image.open(os.path.join(sb_dir_model, img)).convert("L")) / 255.

            control_out = np.asarray(
                Image.open(os.path.join(sb_dir_control, img)).convert("L")) / 255.

            # -------------------------------------------------------------------------------
            # Process Edges
            # -------------------------------------------------------------------------------
            edges_model = (model_out * gt).flatten()
            edges_control = (control_out * gt).flatten()

            for bin_idx in range(len(th_arr) - 1):
                low_th = th_arr[bin_idx]
                high_th = th_arr[bin_idx + 1]

                above_diag, below_diag, on_diag, x_in_bin, y_in_bin = \
                    get_above_below_on_diagonal_counts(edges_control, edges_model, gt.flatten(), low_th, high_th)

                edges_count[bin_idx, 0] += above_diag
                edges_count[bin_idx, 1] += below_diag
                edges_count[bin_idx, 2] += on_diag

                # scat_edges_ax.scatter(x_in_bin, y_in_bin)

            # # ------------------------------
            # # Debug
            # # ------------------------------
            # # Plot
            # f, ax_arr = plt.subplots(1, 3)
            # ax_arr[0].imshow(gt)
            # ax_arr[0].set_title("GT: Input image {}".format(img))
            # ax_arr[1].imshow(model_out)
            # ax_arr[1].set_title("model")
            # ax_arr[2].imshow(control_out)
            # ax_arr[2].set_title("control")
            #
            # # Additions
            # total_counts = edges_count.sum(axis=0)
            #
            # print("Total Counts: A {} [+{}], B {} [+{}]. O {} [={}]".format(
            #     total_counts[0], total_counts[0] - prev_sums[0],
            #     total_counts[1], total_counts[1] - prev_sums[1],
            #     total_counts[2], total_counts[2] - prev_sums[2],
            # ))
            #
            # prev_sums = total_counts
            #
            # import pdb
            # pdb.set_trace()

        list_of_edge_counts.append(edges_count)

        # per bin scatter plots
        # --------------------------
        # scat_edges_ax.plot(x, x, color='red', linewidth=3)
        # scat_edges_ax.set_xlabel("Control")
        # scat_edges_ax.set_ylabel("Model")
        # scat_edges_ax.set_title("Edges Scatter Plot")
        #
        # print("Edges Count {}".format('*' * 80))
        # for bin_idx in range(len(th_arr) - 1):
        #     print("bin [{:0.1f},{:0.1f}]. above {}, below {}, on {}".format(
        #         th_arr[bin_idx], th_arr[bin_idx + 1],
        #         edges_count[bin_idx, 0], edges_count[bin_idx, 1], edges_count[bin_idx, 2]))
        #
        # import pdb
        # pdb.set_trace()

    # -----------------------------------------------------------------------------------
    # Process Results
    # -----------------------------------------------------------------------------------
    edge_counts_sums = np.zeros((len(list_of_sub_dirs), 3))

    for sb_dir_idx, sb_dir in enumerate(list_of_sub_dirs):
        # Edge Count sums
        edge_counts_sums[sb_dir_idx] = list_of_edge_counts[sb_dir_idx].sum(axis=0)

        #  Plot Results per length bins individually
        # ------------------------------------------
        plt.figure()
        plt.plot(th_arr[1:], list_of_edge_counts[sb_dir_idx][:, 0], label='above_diagonal',
                 marker='x', markersize=10)
        plt.plot(th_arr[1:], list_of_edge_counts[sb_dir_idx][:, 1], label='below_diagonal',
                 marker='x', markersize=10)
        plt.plot(th_arr[1:], list_of_edge_counts[sb_dir_idx][:, 2], label='on_diagonal',
                 marker='x', markersize=10)
        plt.xlabel("bin")
        plt.xlim([0, 1])
        plt.ylabel("Count")
        plt.title("Edges {}. Total Diagonal: Above {}, Below {}, On {}".format(
            sb_dir,
            list_of_edge_counts[sb_dir_idx][:, 0].sum(),
            list_of_edge_counts[sb_dir_idx][:, 1].sum(),
            list_of_edge_counts[sb_dir_idx][:, 2].sum()))
        plt.grid()
        plt.legend()

    # -----------------------------------------------------------------------------------
    # PLots of Total Counts
    # -----------------------------------------------------------------------------------
    # [1] Simple Plot
    # ---------------
    plt.figure()
    len_arr = [int(sub_dir.split('_')[1]) for sub_dir in list_of_sub_dirs]
    plt.plot(len_arr, edge_counts_sums[:, 0], label='above_diagonal', marker='x', markersize=10)
    plt.plot(len_arr, edge_counts_sums[:, 1], label='below_diagonal', marker='x', markersize=10)
    plt.plot(len_arr, edge_counts_sums[:, 2], label='on_diagonal', marker='x', markersize=10)
    plt.legend()
    plt.grid()
    plt.xlabel("Contour Length")
    plt.ylabel("Counts")

    # [2] Bar graph
    # -------------
    x = np.arange(len(list_of_sub_dirs))
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    above_bars = ax.bar(x - width, edge_counts_sums[:, 0], width, label='Above')
    below_bars = ax.bar(x, edge_counts_sums[:, 1], width, label='Below')
    on_bars = ax.bar(x + width, edge_counts_sums[:, 2], width, label='On')

    ax.set_xticks(x)
    ax.set_xticklabels(list_of_sub_dirs)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height.
           REF: https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/
           barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        """

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(above_bars)
    autolabel(below_bars)
    autolabel(on_bars)
    ax.set_xlabel("Contour Lengths")
    ax.set_ylabel("Pixel Counts")
    ax.set_title("Total Edge Counts")
    ax.legend()

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    pdb.set_trace()
