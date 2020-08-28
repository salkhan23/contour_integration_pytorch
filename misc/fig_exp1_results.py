# ---------------------------------------------------------------------------------------
# Plot The results fo experiment 1- Contour Gain Experiment
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markersize': 10,
})

# ---------------------------------------------------------------------------------------
# Model Results
# ---------------------------------------------------------------------------------------
model_c_len_iou = np.array([0.9649, 0.4991, 0.8581, 0.8778, 0.8783])
model_c_len_pop_mean_gain = np.array([0.0000, 1.6225, 1.9506, 2.0322, 2.0437])
model_c_len_pop_std_gain = np.array([0.2671, 0.3762, 0.3685, 0.3690, 0.3984])
model_f_spacing_mean_gain = \
    np.array([1.9390, 2.0494, 1.9130, 1.7918, 1.5617, 1.4508, 1.3427, 1.2988])
model_f_spacing_std_gain = \
    np.array([0.3362, 0.3660, 0.3205, 0.2811, 0.2677, 0.2300, 0.2165, 0.1946])

# ---------------------------------------------------------------------------------------
# Control Results
# ---------------------------------------------------------------------------------------
control_c_len_iou = np.array([0.9800, 0.0003, 0.3775, 0.6886, 0.8476])
control_c_len_pop_mean_gain = np.array([0.9999, 0.0000, 0.0000, 0.0000, 0.0162])
control_c_len_pop_std_gain = np.array([0.8910, 0.0000, 0.0000, 0.0000, 0.1132])
control_f_spacing_mean_gain = \
    np.array([0.0198, 0.0164, 0.1844, 0.7491, 1.5870, 1.3039, 1.6905, 1.2291])
control_f_spacing_std_gain = \
    np.array([0.1386, 0.1150, 0.5309, 0.9906, 1.3389, 1.2174, 1.3555, 0.9513])
# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    plt.ion()

    c_len_arr = np.array([1, 3, 5, 7, 9])
    rcd_arr = np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00])

    f, ax_arr = plt.subplots(1, 3, figsize=(15, 5))

    # IOu vs Length
    ax_arr[0].plot(c_len_arr, model_c_len_iou, markersize=10, color='blue', label='model')
    ax_arr[0].plot(c_len_arr, control_c_len_iou, markersize=10, color='red', label='control')

    ax_arr[0].legend()
    ax_arr[0].set_xlabel("Contour Length")
    ax_arr[0].set_ylabel("IoU")
    ax_arr[0].set_xticks(c_len_arr)

    # Gain vs Length
    ax_arr[1].errorbar(
        c_len_arr, model_c_len_pop_mean_gain, model_c_len_pop_std_gain,
        label='Model', capsize=10, capthick=3, markersize=10, color='blue'
    )
    ax_arr[1].errorbar(
        c_len_arr, control_c_len_pop_mean_gain, control_c_len_pop_std_gain,
        label='Control', capsize=10, capthick=3, markersize=10, color='red'
    )
    # ax_arr[1].legend()
    ax_arr[1].set_xlabel("Contour Length")
    ax_arr[1].set_ylabel("Gain")
    ax_arr[1].set_xticks(c_len_arr)

    # Gain vs Fragment Spacing
    ax_arr[2].errorbar(
        rcd_arr, model_f_spacing_mean_gain, model_f_spacing_std_gain,
        label='Model', capsize=10, capthick=3, markersize=10, color='blue'
    )
    ax_arr[2].errorbar(
        rcd_arr, control_f_spacing_mean_gain, control_f_spacing_std_gain,
        label='Control', capsize=10, capthick=3, markersize=10, color='red'
    )
    # ax_arr[2].legend()
    ax_arr[2].set_xlabel("RCD")
    ax_arr[2].set_ylabel("Gain")
    # ax_arr[2].set_xticks(rcd_arr)

# ---------------------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------------------
print("End")
import pdb
pdb.set_trace()


