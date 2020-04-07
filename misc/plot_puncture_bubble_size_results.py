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

results_fwhm_10 = {
    'iou_vs_epoch': np.array([
        [0, 0.2186, 0.0617, 0.1088, 0.4619, 0.001],
        [1, 0.0915, 0.5328, 0.0863, 0.5530, 0.001],
        [2, 0.0812, 0.5771, 0.0787, 0.5807, 0.001],
        [3, 0.0776, 0.5950, 0.0788, 0.5801, 0.001],
        [4, 0.0746, 0.6063, 0.0760, 0.5941, 0.001],
        [5, 0.0727, 0.6135, 0.0738, 0.6042, 0.001],
        [6, 0.0722, 0.6191, 0.0724, 0.6156, 0.001],
        [7, 0.0715, 0.6239, 0.0715, 0.6204, 0.001],
        [8, 0.0702, 0.6277, 0.0711, 0.6208, 0.001],
        [9, 0.0697, 0.6310, 0.0717, 0.6186, 0.001],
        [10, 0.0694, 0.6341, 0.0703, 0.6237, 0.001],
        [11, 0.0686, 0.6367, 0.0700, 0.6279, 0.001],
        [12, 0.0682, 0.6390, 0.0693, 0.6328, 0.001],
        [13, 0.0679, 0.6411, 0.0695, 0.6297, 0.001],
        [14, 0.0676, 0.6429, 0.0691, 0.6338, 0.001],
        [15, 0.0673, 0.6445, 0.0685, 0.6376, 0.001],
        [16, 0.0670, 0.6459, 0.0686, 0.6362, 0.001],
        [17, 0.0668, 0.6474, 0.0684, 0.6391, 0.001],
        [18, 0.0666, 0.6488, 0.0680, 0.6406, 0.001],
        [19, 0.0664, 0.6501, 0.0684, 0.6370, 0.001],
        [20, 0.0661, 0.6512, 0.0683, 0.6378, 0.001],
        [21, 0.0659, 0.6522, 0.0679, 0.6390, 0.001],
        [22, 0.0656, 0.6535, 0.0675, 0.6419, 0.001],
        [23, 0.0655, 0.6544, 0.0673, 0.6428, 0.001],
        [24, 0.0654, 0.6552, 0.0677, 0.6429, 0.001],
        [25, 0.0652, 0.6559, 0.0668, 0.6454, 0.001],
        [26, 0.0651, 0.6567, 0.0664, 0.6481, 0.001],
        [27, 0.0649, 0.6575, 0.0668, 0.6477, 0.001],
        [28, 0.0648, 0.6581, 0.0671, 0.6463, 0.001],
        [29, 0.0647, 0.6587, 0.0671, 0.6461, 0.001],
        [30, 0.0646, 0.6593, 0.0666, 0.6479, 0.001],
        [31, 0.0558, 0.6760, 0.0550, 0.6793, 0.0001],
        [32, 0.0554, 0.6771, 0.0550, 0.6794, 0.0001],
        [33, 0.0553, 0.6776, 0.0549, 0.6796, 0.0001],
        [34, 0.0552, 0.6780, 0.0548, 0.6801, 0.0001],
        [35, 0.0551, 0.6785, 0.0547, 0.6805, 0.0001],
        [36, 0.0550, 0.6789, 0.0546, 0.6809, 0.0001],
        [37, 0.0549, 0.6792, 0.0545, 0.6811, 0.0001],
        [38, 0.0548, 0.6795, 0.0545, 0.6814, 0.0001],
        [39, 0.0548, 0.6798, 0.0544, 0.6816, 0.0001],
        [40, 0.0547, 0.6801, 0.0543, 0.6819, 0.0001],
        [41, 0.0546, 0.6803, 0.0543, 0.6821, 0.0001],
        [42, 0.0546, 0.6806, 0.0542, 0.6824, 0.0001],
        [43, 0.0545, 0.6808, 0.0542, 0.6826, 0.0001],
        [44, 0.0545, 0.6810, 0.0541, 0.6829, 0.0001],
        [45, 0.0544, 0.6812, 0.0541, 0.6830, 0.0001],
        [46, 0.0544, 0.6814, 0.0540, 0.6834, 0.0001],
        [47, 0.0543, 0.6816, 0.0539, 0.6837, 0.0001],
        [48, 0.0543, 0.6818, 0.0539, 0.6838, 0.0001],
        [49, 0.0542, 0.6820, 0.0538, 0.6841, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0018, 0.9993, 0.9984, 1.0026]),
        'std_gain': np.array([0.1212, 0.1248, 0.0895, 0.0741, 0.1007])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([0.9972, 1.0016, 1.0057, 1.0036, 1.0039, 1.0049, 1.0039, 1.0041]),
        'std_gain': np.array([0.0509, 0.0373, 0.0422, 0.0245, 0.0206, 0.0118, 0.0134, 0.0098])
    }
}

results_fwhm_20 = {
    'iou_vs_epoch': np.array([
        [0, 0.2746, 0.0941, 0.1311, 0.3437, 0.001],
        [1, 0.1175, 0.4083, 0.1127, 0.4594, 0.001],
        [2, 0.1065, 0.4543, 0.1061, 0.4774, 0.001],
        [3, 0.1020, 0.4762, 0.1042, 0.4828, 0.001],
        [4, 0.0986, 0.4896, 0.0988, 0.5022, 0.001],
        [5, 0.0969, 0.4987, 0.0995, 0.5096, 0.001],
        [6, 0.0955, 0.5068, 0.0973, 0.5136, 0.001],
        [7, 0.0937, 0.5134, 0.0946, 0.5181, 0.001],
        [8, 0.0927, 0.5181, 0.0938, 0.5231, 0.001],
        [9, 0.0911, 0.5225, 0.0967, 0.5275, 0.001],
        [10, 0.0923, 0.5271, 0.0922, 0.5360, 0.001],
        [11, 0.0900, 0.5306, 0.0913, 0.5383, 0.001],
        [12, 0.0887, 0.5339, 0.0899, 0.5372, 0.001],
        [13, 0.0879, 0.5370, 0.0895, 0.5377, 0.001],
        [14, 0.0876, 0.5396, 0.0899, 0.5431, 0.001],
        [15, 0.0877, 0.5412, 0.0895, 0.5427, 0.001],
        [16, 0.0866, 0.5441, 0.0898, 0.5416, 0.001],
        [17, 0.0862, 0.5461, 0.0887, 0.5441, 0.001],
        [18, 0.0860, 0.5478, 0.0885, 0.5444, 0.001],
        [19, 0.0859, 0.5494, 0.0881, 0.5466, 0.001],
        [20, 0.0858, 0.5511, 0.0884, 0.5508, 0.001],
        [21, 0.0853, 0.5525, 0.0869, 0.5497, 0.001],
        [22, 0.0848, 0.5542, 0.0869, 0.5492, 0.001],
        [23, 0.0845, 0.5557, 0.0862, 0.5516, 0.001],
        [24, 0.0843, 0.5568, 0.0854, 0.5570, 0.001],
        [25, 0.0842, 0.5576, 0.0858, 0.5554, 0.001],
        [26, 0.0839, 0.5589, 0.0852, 0.5565, 0.001],
        [27, 0.0837, 0.5600, 0.0849, 0.5585, 0.001],
        [28, 0.0840, 0.5605, 0.0843, 0.5599, 0.001],
        [29, 0.0834, 0.5616, 0.0842, 0.5627, 0.001],
        [30, 0.0832, 0.5625, 0.0845, 0.5637, 0.001],
        [31, 0.0741, 0.5795, 0.0734, 0.5818, 0.0001],
        [32, 0.0737, 0.5807, 0.0736, 0.5824, 0.0001],
        [33, 0.0736, 0.5812, 0.0736, 0.5832, 0.0001],
        [34, 0.0734, 0.5818, 0.0735, 0.5839, 0.0001],
        [35, 0.0733, 0.5822, 0.0734, 0.5846, 0.0001],
        [36, 0.0732, 0.5827, 0.0733, 0.5851, 0.0001],
        [37, 0.0731, 0.5831, 0.0732, 0.5855, 0.0001],
        [38, 0.0730, 0.5834, 0.0731, 0.5861, 0.0001],
        [39, 0.0729, 0.5838, 0.0730, 0.5865, 0.0001],
        [40, 0.0729, 0.5841, 0.0729, 0.5868, 0.0001],
        [41, 0.0728, 0.5844, 0.0728, 0.5871, 0.0001],
        [42, 0.0727, 0.5847, 0.0727, 0.5874, 0.0001],
        [43, 0.0726, 0.5850, 0.0726, 0.5878, 0.0001],
        [44, 0.0726, 0.5852, 0.0725, 0.5882, 0.0001],
        [45, 0.0725, 0.5855, 0.0725, 0.5885, 0.0001],
        [46, 0.0725, 0.5858, 0.0724, 0.5887, 0.0001],
        [47, 0.0724, 0.5860, 0.0723, 0.5890, 0.0001],
        [48, 0.0723, 0.5862, 0.0723, 0.5893, 0.0001],
        [49, 0.0723, 0.5865, 0.0722, 0.5895, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0223, 1.0216, 1.0413, 1.0296]),
        'std_gain': np.array([0.2973, 0.2158, 0.2313, 0.2743, 0.2827])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0152, 1.0437, 1.0421, 1.0407, 1.0484, 1.0546, 1.0601, 1.0739]),
        'std_gain': np.array([0.1914, 0.2112, 0.1663, 0.1910, 0.1469, 0.1420, 0.1090, 0.1143])
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
    plot_iou_results(results_fwhm_10, ax=axis, label='fwhm_10', c='r')
    plot_iou_results(results_fwhm_20, ax=axis, label='fwhm_20', c='g')

    plt.legend()
    plt.title("IoU vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid()

    # Contour Length
    _, axis = plt.subplots()

    plot_gain_vs_contour_len(results_fwhm_5, ax=axis, label='fwhm_5', c='b')
    plot_gain_vs_contour_len(results_fwhm_10, ax=axis, label='fwhm_10', c='r')
    plot_gain_vs_contour_len(results_fwhm_20, ax=axis, label='fwhm_20', c='g')

    plt.legend()
    plt.title("Contour Gain vs Length")
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # Fragment Spacing
    _, axis = plt.subplots()

    plot_gain_vs_fragment_spacing(results_fwhm_5, ax=axis, label='fwhm_5', c='b')
    plot_gain_vs_fragment_spacing(results_fwhm_10, ax=axis, label='fwhm_10', c='r')
    plot_gain_vs_fragment_spacing(results_fwhm_20, ax=axis, label='fwhm_20', c='g')

    plt.legend()
    plt.title("Contour Gain vs Fragment Spacing")
    plt.xlabel("Fragment Spacing (RCD)")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # -----------------------------------------------------------------------------------
    import pdb

    pdb.set_trace()
