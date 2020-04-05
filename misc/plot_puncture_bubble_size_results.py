# ---------------------------------------------------------------------------------------
# Plot results for the puncture bubble size explorer experiment
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

n_bubbles = 10


mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)


results_fwhm_5 = {
    'iou_vs_epoch': np.array([
        [0, 0.2247, 0.2367, 0.1311, 0.3734, 0.001],
        [1, 0.0876, 0.5694, 0.0802, 0.6107, 0.001],
        [2, 0.0728, 0.6219, 0.0692, 0.6397, 0.001],
        [3, 0.0681, 0.6419, 0.0660, 0.6480, 0.001],
        [4, 0.0651, 0.6544, 0.0652, 0.6530, 0.001],
        [5, 0.0642, 0.6624, 0.0667, 0.6545, 0.001],
        [6, 0.0625, 0.6684, 0.0619, 0.6688, 0.001],
        [7, 0.0615, 0.6730, 0.0619, 0.6668, 0.001],
        [8, 0.0608, 0.6765, 0.0616, 0.6688, 0.001],
        [9, 0.0602, 0.6795, 0.0608, 0.6751, 0.001],
        [10, 0.0597, 0.6824, 0.0602, 0.6796, 0.001],
        [11, 0.0593, 0.6847, 0.0597, 0.6803, 0.001],
        [12, 0.0590, 0.6867, 0.0590, 0.6846, 0.001],
        [13, 0.0586, 0.6885, 0.0585, 0.6874, 0.001],
        [14, 0.0586, 0.6898, 0.0587, 0.6851, 0.001],
        [15, 0.0582, 0.6913, 0.0581, 0.6889, 0.001],
        [16, 0.0580, 0.6926, 0.0576, 0.6910, 0.001],
        [17, 0.0577, 0.6937, 0.0577, 0.6913, 0.001],
        [18, 0.0576, 0.6947, 0.0574, 0.6930, 0.001],
        [19, 0.0574, 0.6955, 0.0574, 0.6937, 0.001],
        [20, 0.0573, 0.6965, 0.0572, 0.6954, 0.001],
        [21, 0.0571, 0.6973, 0.0570, 0.6964, 0.001],
        [22, 0.0570, 0.6981, 0.0568, 0.6975, 0.001],
        [23, 0.0568, 0.6988, 0.0567, 0.6980, 0.001],
        [24, 0.0568, 0.6995, 0.0566, 0.6983, 0.001],
        [25, 0.0566, 0.7002, 0.0564, 0.7004, 0.001],
        [26, 0.0565, 0.7010, 0.0564, 0.6996, 0.001],
        [27, 0.0564, 0.7017, 0.0564, 0.7002, 0.001],
        [28, 0.0562, 0.7024, 0.0560, 0.7032, 0.001],
        [29, 0.0561, 0.7031, 0.0558, 0.7040, 0.001],
        [30, 0.0560, 0.7037, 0.0560, 0.7041, 0.001],
        [31, 0.0482, 0.7186, 0.0474, 0.7232, 0.0001],
        [32, 0.0480, 0.7193, 0.0473, 0.7241, 0.0001],
        [33, 0.0479, 0.7197, 0.0473, 0.7244, 0.0001],
        [34, 0.0478, 0.7200, 0.0473, 0.7246, 0.0001],
        [35, 0.0478, 0.7204, 0.0472, 0.7248, 0.0001],
        [36, 0.0477, 0.7207, 0.0472, 0.7250, 0.0001],
        [37, 0.0476, 0.7209, 0.0471, 0.7253, 0.0001],
        [38, 0.0476, 0.7212, 0.0471, 0.7255, 0.0001],
        [39, 0.0475, 0.7215, 0.0470, 0.7260, 0.0001],
        [40, 0.0475, 0.7218, 0.0470, 0.7263, 0.0001],
        [41, 0.0474, 0.7220, 0.0469, 0.7265, 0.0001],
        [42, 0.0474, 0.7222, 0.0468, 0.7269, 0.0001],
        [43, 0.0473, 0.7225, 0.0468, 0.7273, 0.0001],
        [44, 0.0473, 0.7227, 0.0467, 0.7276, 0.0001],
        [45, 0.0472, 0.7229, 0.0467, 0.7279, 0.0001],
        [46, 0.0472, 0.7231, 0.0466, 0.7283, 0.0001],
        [47, 0.0471, 0.7233, 0.0466, 0.7285, 0.0001],
        [48, 0.0471, 0.7235, 0.0465, 0.7288, 0.0001],
        [49, 0.0470, 0.7237, 0.0465, 0.7290, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0063, 1.0060, 1.0051, 1.0066]),
        'std_gain': np.array([0.0826, 0.0605, 0.0617, 0.0591, 0.0537])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0082, 1.0102, 1.0094, 1.0100, 1.0116, 1.0145, 1.0155, 1.0172]),
        'std_gain': np.array([0.0587, 0.0385, 0.0452, 0.0266, 0.0249, 0.0180, 0.0130, 0.0111])
    }
}


def plot_iou_results(results_dict, ax=None, label='', c='k'):
    if axis is None:
        f, ax = plt.subplots()

    ax.plot(
        results_dict['iou_vs_epoch'][:, 0],
        results_dict['iou_vs_epoch'][:, 2],
        label='train_' + label,
        linestyle='--',
        color=c
    )

    ax.plot(
        results_dict['iou_vs_epoch'][:, 0],
        results_dict['iou_vs_epoch'][:, 4],
        label='val_' + label,
        linestyle='-',
        color=c
    )


def plot_gain_vs_contour_len(results_dict, ax=None, label='', c='k'):
    if axis is None:
        f, ax = plt.subplots()

    ax.errorbar(
        results_dict['gain_vs_c_len']['c_len'],
        results_dict['gain_vs_c_len']['mean_gain'],
        results_dict['gain_vs_c_len']['std_gain'],
        label=label,
        linestyle='--',
        color=c
    )


def plot_gain_vs_fragment_spacing(results_dict, ax=None, label='', c='k'):
    if axis is None:
        f, ax = plt.subplots()

    ax.errorbar(
        results_dict['gain_vs_spacing']['spacing'],
        results_dict['gain_vs_spacing']['mean_gain'],
        results_dict['gain_vs_spacing']['std_gain'],
        label=label,
        linestyle='--',
        color=c
    )


if __name__ == "__main__":
    plt.ion()

    _, axis = plt.subplots()

    plot_iou_results(results_fwhm_5, ax=axis, label='fwhm_5', c='b')

    plt.legend()
    plt.title("IoU vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid()

    # Contour Length
    _, axis = plt.subplots()

    plot_gain_vs_contour_len(results_fwhm_5, ax=axis, label='fwhm_5', c='b')

    plt.legend()
    plt.title("Contour Gain vs Length")
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # Fragment Spacing
    _, axis = plt.subplots()

    plot_gain_vs_fragment_spacing(results_fwhm_5, ax=axis, label='fwhm_5', c='b')

    plt.legend()
    plt.title("Contour Gain vs Fragment Spacing")
    plt.xlabel("Fragment Spacing (RCD)")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
