# ---------------------------------------------------------------------------------------
# Plot Results of Classification task on Imagenet.
# Data below is for a contour/control model embedded into a pretrained resnet 50 model
# ---------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)

# Epoch, train_loss, train_accTop1, train_accTop5, val_loss val_accTop1, val_accTop5
resnet50 = np.array([
    [0, np.nan,  np.nan,  np.nan, 0.9618, 76.1300, 92.8620],
    [1, 1.3698, 67.3453, 86.7399, 1.1767, 71.1780, 90.3280],
    [2, 1.3757, 67.2907, 86.6647, 1.1728, 71.1800, 90.3480],
    [3, 1.3712, 67.4470, 86.7218, 1.1589, 71.1160, 90.5560],
    [4, 1.3611, 67.6114, 86.8233, 1.1931, 70.7420, 90.0280],
    [5, 1.3583, 67.7314, 86.8291, 1.1571, 71.6480, 90.6240],
    [6, 1.3521, 67.8774, 86.9465, 1.1550, 71.6960, 90.5500],
    [7, 1.3469, 67.9887, 87.0224, 1.1418, 71.7940, 90.7180],
    [8, 1.3438, 68.0303, 87.0329, 1.1426, 71.5340, 90.7680],
    [9, 1.3408, 68.1176, 87.0912, 1.1511, 71.4380, 90.7520],
    [10, 1.3375, 68.1943, 87.1412, 1.1313, 72.0100, 90.8920],
    [11, 1.3316, 68.3022, 87.2274, 1.1337, 72.1380, 90.8600],
    [12, 1.3294, 68.3686, 87.2512, 1.1431, 71.8180, 90.6740],
])

resnet50_with_contour_integration = np.array([

    [0, np.nan,  np.nan,  np.nan, 9.1885,  5.3960, 14.5360],
    [1, 0.9970, 75.8740, 91.2289, 0.9644, 76.1420, 92.8840],
    [2, 0.9852, 76.1348, 91.3521, 0.9652, 76.0880, 92.9380],
    [3, 0.9846, 76.1604, 91.3488, 0.9669, 76.0760, 92.8120],
    [4, 0.9835, 76.1564, 91.3855, 0.9681, 76.1040, 92.8920],
    [5, 0.9828, 76.1746, 91.3989, 0.9626, 76.0780, 92.9260],
    [6, 0.9831, 76.1763, 91.4111, 0.9609, 76.1220, 92.8700],
    [7, 0.9830, 76.1814, 91.3893, 0.9628, 76.1900, 92.8400],
    [8, 0.9814, 76.2162, 91.3828, 0.9634, 76.2580, 92.9120],
    [9, 0.9848, 76.1574, 91.3315, 0.9628, 76.2480, 92.9140],
    [10, 0.9834, 76.2140, 91.3673, 0.9634, 76.1540, 92.8480],
    [11, 0.9809, 76.2368, 91.3963, 0.9684, 76.1880, 92.9060],
    [12, 0.9808, 76.1958, 91.4374, 0.9630, 76.1460, 92.8500],
    [13, 0.9802, 76.2162, 91.4125, 0.9624, 76.1700, 92.8480],
    [14, 0.9808, 76.2367, 91.4076, 0.9621, 76.1740, 92.9140],
])

resnet50_with_control = np.array([
    [0, np.nan,  np.nan,  np.nan, 9.6688,  0.1520,  0.6540],
    [1, 1.1965, 71.7518, 88.8480, 2.0599, 55.4600, 78.8860],
    [2, 1.0894, 73.8884, 90.2064, 2.1085, 54.9300, 78.1120],
    [3, 1.0720, 74.3006, 90.3867, 2.2696, 52.2500, 75.9620],
    [4, 1.0659, 74.3860, 90.4581, 2.1042, 54.9720, 78.2140],
    [5, 1.0594, 74.5742, 90.5209, 2.1431, 54.1360, 77.5960],
    [6, 1.0533, 74.6656, 90.6033, 2.2456, 52.5800, 76.3280],
    [7, 1.0510, 74.6937, 90.6432, 2.3276, 51.5640, 75.3100],
    [8, 1.0509, 74.7202, 90.6488, 2.2458, 52.7040, 76.3200],
    [9, 1.0457, 74.8495, 90.7058, 2.2262, 52.8340, 76.4700],
    [10, 1.0455, 74.8384, 90.6821, 2.3330, 51.5240, 75.0880],
    [11, 1.0464, 74.8088, 90.6762, 2.1380, 54.2260, 77.5760],
    [12, 1.0413, 74.9616, 90.7462, 2.2782, 52.3440, 75.7960],
])


# ---------------------------------------------------------------------------------------
plt.ion()

f, ax_arr = plt.subplots(2, 1, sharex=True)
ax_arr[0].plot(resnet50[:, 2], label='train - resnet50')
ax_arr[0].plot(resnet50_with_contour_integration[:, 2], label='+ contour integration')
ax_arr[0].plot(resnet50_with_control[:, 2], label='+ control')
ax_arr[0].set_title("Train")
ax_arr[0].legend()
ax_arr[0].set_ylabel("Accuracy")
ax_arr[0].grid()

ax_arr[1].plot(resnet50[:, 5], label='val - resnet50')
ax_arr[1].plot(resnet50_with_contour_integration[:, 5], label='+ contour integration')
ax_arr[1].plot(resnet50_with_control[:, 5], label='val + control')
ax_arr[1].axhline((100 - 23.85), linestyle='--', color='black')
ax_arr[1].text(0, (100 - 23.85), 'Published', color='black')
ax_arr[1].set_xlabel("Epoch")
ax_arr[1].set_ylabel("Accuracy")
ax_arr[1].set_title("Validation")
ax_arr[1].legend()
ax_arr[1].grid()

f.suptitle("Top1 Accuracy")


f, ax_arr = plt.subplots(2, 1, sharex=True)
ax_arr[0].plot(resnet50[:, 3], label='resnet50')
ax_arr[0].plot(resnet50_with_contour_integration[:, 3], label='+ contour integration')
ax_arr[0].plot(resnet50_with_control[:, 3], label='+ control')
ax_arr[0].set_title("Train")
ax_arr[0].legend()
ax_arr[0].set_ylabel("Accuracy")
ax_arr[0].grid()

ax_arr[1].plot(resnet50[:, 6], label='resnet50')
ax_arr[1].plot(resnet50_with_contour_integration[:, 6], label='+ contour control')
ax_arr[1].plot(resnet50_with_control[:, 6], label='+ control')
ax_arr[1].axhline((100 - 7.13), linestyle='--', color='black')
ax_arr[1].text(0, (100 - 7.13), 'Published', color='black')
ax_arr[1].set_xlabel("Epoch")
ax_arr[1].set_ylabel("Accuracy")
ax_arr[1].set_title("Validation")
ax_arr[1].legend()
ax_arr[1].grid()

f.suptitle("Top5 Accuracy")

input("Press any key to exit")


