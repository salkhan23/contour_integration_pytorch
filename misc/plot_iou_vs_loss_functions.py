# ---------------------------------------------------------------------------------------
# Plot the training loss and IoU scores for different loss functions.

# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)

# Default.
# ./results/contour_dataset_sensitivity_analysis/num_iterations/n_iters_5
bce_results = np.array([
    [1, 0.5135, 0.1630, 0.4190, 0.1972, 0.0001],
    [2, 0.3729, 0.1993, 0.3161, 0.1980, 0.0001],
    [3, 0.2542, 0.2008, 0.2199, 0.1988, 0.0001],
    [4, 0.2136, 0.2087, 0.2047, 0.2069, 0.0001],
    [5, 0.2020, 0.2182, 0.1978, 0.2061, 0.0001],
    [6, 0.1984, 0.2236, 0.1993, 0.2161, 0.0001],
    [7, 0.1946, 0.2266, 0.1937, 0.2074, 0.0001],
    [8, 0.1899, 0.2389, 0.1894, 0.2476, 0.0001],
    [9, 0.1867, 0.2547, 0.1858, 0.2708, 0.0001],
    [10, 0.1841, 0.2754, 0.1860, 0.2356, 0.0001],
    [11, 0.1819, 0.3002, 0.1806, 0.2934, 0.0001],
    [12, 0.1796, 0.3257, 0.1800, 0.3585, 0.0001],
    [13, 0.1777, 0.3523, 0.1762, 0.3491, 0.0001],
    [14, 0.1759, 0.3746, 0.1757, 0.4070, 0.0001],
    [15, 0.1741, 0.4009, 0.1732, 0.4213, 0.0001],
    [16, 0.1726, 0.4232, 0.1746, 0.4188, 0.0001],
    [17, 0.1715, 0.4369, 0.1747, 0.4761, 0.0001],
    [18, 0.1701, 0.4585, 0.1708, 0.4192, 0.0001],
    [19, 0.1692, 0.4711, 0.1686, 0.4808, 0.0001],
    [20, 0.1681, 0.4847, 0.1680, 0.4913, 0.0001],
    [21, 0.1671, 0.4971, 0.1717, 0.4982, 0.0001],
    [22, 0.1663, 0.5093, 0.1719, 0.4028, 0.0001],
    [23, 0.1655, 0.5218, 0.1669, 0.4944, 0.0001],
    [24, 0.1646, 0.5352, 0.1650, 0.5490, 0.0001],
    [25, 0.1641, 0.5427, 0.1654, 0.5239, 0.0001],
    [26, 0.1635, 0.5536, 0.1659, 0.5678, 0.0001],
    [27, 0.1628, 0.5627, 0.1747, 0.5094, 0.0001],
    [28, 0.1623, 0.5718, 0.1654, 0.5149, 0.0001],
    [29, 0.1618, 0.5816, 0.1642, 0.5593, 0.0001],
    [30, 0.1615, 0.5869, 0.1646, 0.5460, 0.0001],
    [31, 0.1609, 0.5960, 0.1639, 0.5977, 0.0001],
    [32, 0.1607, 0.6000, 0.1650, 0.5362, 0.0001],
    [33, 0.1601, 0.6116, 0.1609, 0.5986, 0.0001],
    [34, 0.1599, 0.6145, 0.1607, 0.6157, 0.0001],
    [35, 0.1595, 0.6225, 0.1615, 0.6089, 0.0001],
    [36, 0.1591, 0.6292, 0.1599, 0.6162, 0.0001],
    [37, 0.1589, 0.6344, 0.1609, 0.6156, 0.0001],
    [38, 0.1586, 0.6391, 0.1604, 0.6081, 0.0001],
    [39, 0.1584, 0.6432, 0.1600, 0.6409, 0.0001],
    [40, 0.1581, 0.6504, 0.1591, 0.6402, 0.0001],
    [41, 0.1579, 0.6543, 0.1593, 0.6379, 0.0001],
    [42, 0.1576, 0.6598, 0.1624, 0.6067, 0.0001],
    [43, 0.1573, 0.6652, 0.1595, 0.6524, 0.0001],
    [44, 0.1571, 0.6690, 0.1587, 0.6594, 0.0001],
    [45, 0.1569, 0.6738, 0.1602, 0.6228, 0.0001],
    [46, 0.1567, 0.6786, 0.1587, 0.6518, 0.0001],
    [47, 0.1566, 0.6808, 0.1595, 0.6306, 0.0001],
    [48, 0.1564, 0.6854, 0.1599, 0.6360, 0.0001],
    [49, 0.1562, 0.6883, 0.1573, 0.6872, 0.0001],
    [50, 0.1560, 0.6928, 0.1582, 0.6734, 0.0001],
    [51, 0.1560, 0.6956, 0.1575, 0.6813, 0.0001],
    [52, 0.1558, 0.6985, 0.1575, 0.6821, 0.0001],
    [53, 0.1555, 0.7044, 0.1591, 0.6622, 0.0001],
    [54, 0.1554, 0.7088, 0.1617, 0.6237, 0.0001],
    [55, 0.1553, 0.7100, 0.1573, 0.6744, 0.0001],
    [56, 0.1551, 0.7139, 0.1576, 0.6935, 0.0001],
    [57, 0.1550, 0.7177, 0.1572, 0.6934, 0.0001],
    [58, 0.1549, 0.7207, 0.1570, 0.6856, 0.0001],
    [59, 0.1548, 0.7225, 0.1573, 0.6874, 0.0001],
    [60, 0.1547, 0.7246, 0.1570, 0.6881, 0.0001],
    [61, 0.1546, 0.7287, 0.1589, 0.6581, 0.0001],
    [62, 0.1544, 0.7322, 0.1575, 0.6893, 0.0001],
    [63, 0.1543, 0.7327, 0.1582, 0.6820, 0.0001],
    [64, 0.1543, 0.7333, 0.1596, 0.6566, 0.0001],
    [65, 0.1541, 0.7396, 0.1577, 0.6861, 0.0001],
    [66, 0.1541, 0.7398, 0.1571, 0.6953, 0.0001],
    [67, 0.1541, 0.7404, 0.1577, 0.6839, 0.0001],
    [68, 0.1539, 0.7439, 0.1571, 0.7009, 0.0001],
    [69, 0.1537, 0.7485, 0.1567, 0.7005, 0.0001],
    [70, 0.1537, 0.7514, 0.1562, 0.7083, 0.0001],
    [71, 0.1536, 0.7509, 0.1587, 0.6718, 0.0001],
    [72, 0.1536, 0.7524, 0.1570, 0.7042, 0.0001],
    [73, 0.1535, 0.7552, 0.1565, 0.7229, 0.0001],
    [74, 0.1535, 0.7544, 0.1561, 0.7204, 0.0001],
    [75, 0.1534, 0.7589, 0.1561, 0.7199, 0.0001],
    [76, 0.1533, 0.7599, 0.1582, 0.6937, 0.0001],
    [77, 0.1532, 0.7637, 0.1559, 0.7213, 0.0001],
    [78, 0.1531, 0.7651, 0.1566, 0.7136, 0.0001],
    [79, 0.1531, 0.7655, 0.1568, 0.7115, 0.0001],
    [80, 0.1530, 0.7696, 0.1561, 0.7186, 0.0001],
    [81, 0.1510, 0.7945, 0.1543, 0.7432, 5e-05],
    [82, 0.1508, 0.7953, 0.1543, 0.7290, 5e-05],
    [83, 0.1507, 0.7972, 0.1551, 0.7231, 5e-05],
    [84, 0.1507, 0.7979, 0.1545, 0.7378, 5e-05],
    [85, 0.1506, 0.8003, 0.1550, 0.7247, 5e-05],
    [86, 0.1507, 0.7991, 0.1548, 0.7303, 5e-05],
    [87, 0.1506, 0.8019, 0.1548, 0.7391, 5e-05],
    [88, 0.1505, 0.8024, 0.1544, 0.7393, 5e-05],
    [89, 0.1504, 0.8047, 0.1545, 0.7421, 5e-05],
    [90, 0.1505, 0.8042, 0.1547, 0.7411, 5e-05],
    [91, 0.1504, 0.8062, 0.1545, 0.7454, 5e-05],
    [92, 0.1504, 0.8064, 0.1541, 0.7484, 5e-05],
    [93, 0.1503, 0.8095, 0.1541, 0.7450, 5e-05],
    [94, 0.1503, 0.8098, 0.1549, 0.7328, 5e-05],
    [95, 0.1503, 0.8104, 0.1552, 0.7220, 5e-05],
    [96, 0.1503, 0.8106, 0.1546, 0.7442, 5e-05],
    [97, 0.1503, 0.8119, 0.1550, 0.7331, 5e-05],
    [98, 0.1502, 0.8137, 0.1546, 0.7458, 5e-05],
    [99, 0.1501, 0.8151, 0.1549, 0.7509, 5e-05],
    [100, 0.1502, 0.8142, 0.1549, 0.7361, 5e-05],
])

# ./results/contour_dataset_sensitivity_analysis/loss_functions/class_balanced_bce/
# ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20210506_155054_lr_1e-3_best
class_balanced_bce_results = np.array([
        [1, 93.9072, 0.1945, 53.5722, 0.2065, 0.001],
        [2, 51.4281, 0.2015, 50.5115, 0.2060, 0.001],
        [3, 50.2602, 0.2015, 50.0463, 0.2065, 0.001],
        [4, 49.9501, 0.2015, 49.9799, 0.2060, 0.001],
        [5, 49.8370, 0.2015, 49.8184, 0.2055, 0.001],
        [6, 49.8327, 0.2015, 49.8792, 0.2060, 0.001],
        [7, 49.7943, 0.2015, 49.7660, 0.2060, 0.001],
        [8, 49.7863, 0.2015, 49.7496, 0.2065, 0.001],
        [9, 49.7354, 0.2015, 49.7200, 0.2065, 0.001],
        [10, 49.7176, 0.2015, 49.7143, 0.2060, 0.001],
        [11, 49.6888, 0.2015, 49.6874, 0.2050, 0.001],
        [12, 49.6571, 0.2015, 49.6454, 0.2040, 0.001],
        [13, 49.6333, 0.2015, 49.6403, 0.2060, 0.001],
        [14, 49.6111, 0.2017, 49.6276, 0.2072, 0.001],
        [15, 49.5975, 0.2021, 49.6351, 0.2050, 0.001],
        [16, 49.5837, 0.2015, 49.5939, 0.2055, 0.001],
        [17, 49.5752, 0.2015, 49.5869, 0.2050, 0.001],
        [18, 49.5644, 0.2015, 49.6676, 0.2070, 0.001],
        [19, 49.5593, 0.2016, 49.5748, 0.2060, 0.001],
        [20, 49.5528, 0.2094, 49.5927, 0.2416, 0.001],
        [21, 49.5479, 0.2626, 49.5692, 0.2369, 0.001],
        [22, 49.5608, 0.2699, 49.5797, 0.2811, 0.001],
        [23, 49.5463, 0.3054, 49.5861, 0.2763, 0.001],
        [24, 49.5418, 0.3213, 49.5788, 0.3023, 0.001],
        [25, 49.5383, 0.3380, 49.5658, 0.3174, 0.001],
        [26, 49.5331, 0.3560, 49.6448, 0.3624, 0.001],
        [27, 49.5492, 0.3515, 49.5616, 0.3636, 0.001],
        [28, 49.5265, 0.3832, 49.5749, 0.3220, 0.001],
        [29, 49.5290, 0.3921, 49.5751, 0.3286, 0.001],
        [30, 49.5265, 0.4015, 49.6072, 0.3248, 0.001],
        [31, 49.5180, 0.4205, 49.6113, 0.4180, 0.001],
        [32, 49.5295, 0.4129, 49.5748, 0.3433, 0.001],
        [33, 49.5177, 0.4333, 49.5655, 0.3646, 0.001],
        [34, 49.5131, 0.4472, 49.5572, 0.4480, 0.001],
        [35, 49.5230, 0.4402, 49.5701, 0.4932, 0.001],
        [36, 49.5112, 0.4694, 49.5555, 0.4463, 0.001],
        [37, 49.5122, 0.4683, 49.5887, 0.5072, 0.001],
        [38, 49.5105, 0.4770, 49.5840, 0.3967, 0.001],
        [39, 49.5179, 0.4741, 49.5548, 0.4404, 0.001],
        [40, 49.5222, 0.4681, 49.6188, 0.5090, 0.001],
        [41, 49.5031, 0.5016, 49.5933, 0.5214, 0.001],
        [42, 49.5097, 0.4961, 49.5763, 0.4410, 0.001],
        [43, 49.5102, 0.5000, 49.5974, 0.5005, 0.001],
        [44, 49.5058, 0.5130, 49.5742, 0.4490, 0.001],
        [45, 49.5017, 0.5220, 50.1188, 0.2343, 0.001],
        [46, 49.5177, 0.5014, 49.5732, 0.5182, 0.001],
        [47, 49.4996, 0.5313, 49.6008, 0.4294, 0.001],
        [48, 49.5015, 0.5325, 49.7053, 0.3030, 0.001],
        [49, 49.5049, 0.5258, 49.5767, 0.5426, 0.001],
        [50, 49.4925, 0.5543, 49.5964, 0.4785, 0.001],
        [51, 49.5051, 0.5377, 49.5754, 0.5376, 0.001],
        [52, 49.5024, 0.5470, 49.5892, 0.4420, 0.001],
        [53, 49.5071, 0.5404, 49.5844, 0.5062, 0.001],
        [54, 49.4942, 0.5632, 49.5886, 0.5327, 0.001],
        [55, 49.4987, 0.5597, 49.5707, 0.5288, 0.001],
        [56, 49.4996, 0.5639, 49.6586, 0.5642, 0.001],
        [57, 49.5014, 0.5634, 49.6039, 0.5583, 0.001],
        [58, 49.5018, 0.5644, 49.5789, 0.5260, 0.001],
        [59, 49.5012, 0.5688, 49.5903, 0.4953, 0.001],
        [60, 49.4970, 0.5770, 49.5659, 0.5374, 0.001],
        [61, 49.4934, 0.5862, 49.6152, 0.5647, 0.001],
        [62, 49.4947, 0.5881, 49.6375, 0.5008, 0.001],
        [63, 49.5095, 0.5717, 49.5961, 0.5617, 0.001],
        [64, 49.4867, 0.6049, 49.6069, 0.5267, 0.001],
        [65, 49.4966, 0.5948, 49.5972, 0.5545, 0.001],
        [66, 49.5015, 0.5888, 49.6097, 0.5816, 0.001],
        [67, 49.5014, 0.5877, 49.5843, 0.5282, 0.001],
        [68, 49.4971, 0.5989, 49.6666, 0.5960, 0.001],
        [69, 49.4956, 0.6056, 49.5995, 0.5235, 0.001],
        [70, 49.4903, 0.6116, 49.6222, 0.6247, 0.001],
        [71, 49.4879, 0.6194, 49.5758, 0.5386, 0.001],
        [72, 49.4947, 0.6142, 49.6246, 0.5880, 0.001],
        [73, 49.4920, 0.6197, 49.5874, 0.5250, 0.001],
        [74, 49.4920, 0.6241, 49.6389, 0.6077, 0.001],
        [75, 49.4998, 0.6161, 49.6376, 0.5539, 0.001],
        [76, 49.4906, 0.6262, 49.6968, 0.6479, 0.001],
        [77, 49.4959, 0.6205, 49.6428, 0.4967, 0.001],
        [78, 49.4850, 0.6381, 49.6258, 0.6143, 0.001],
        [79, 49.4972, 0.6258, 49.6388, 0.5568, 0.001],
        [80, 49.4901, 0.6357, 49.6007, 0.5530, 0.001],
        [81, 49.4478, 0.6934, 49.5999, 0.6467, 0.0005],
        [82, 49.4315, 0.7171, 49.5962, 0.6309, 0.0005],
        [83, 49.4295, 0.7247, 49.6672, 0.6848, 0.0005],
        [84, 49.4405, 0.7167, 49.6292, 0.6607, 0.0005],
        [85, 49.4348, 0.7265, 49.6532, 0.6761, 0.0005],
        [86, 49.4372, 0.7269, 49.6419, 0.6523, 0.0005],
        [87, 49.4298, 0.7359, 49.7190, 0.7010, 0.0005],
        [88, 49.4302, 0.7414, 49.7364, 0.6603, 0.0005],
        [89, 49.4494, 0.7169, 49.6485, 0.6752, 0.0005],
        [90, 49.4243, 0.7481, 49.6401, 0.6738, 0.0005],
        [91, 49.4253, 0.7518, 49.6397, 0.6376, 0.0005],
        [92, 49.4390, 0.7368, 49.6664, 0.6466, 0.0005],
        [93, 49.4363, 0.7423, 49.6551, 0.6627, 0.0005],
        [94, 49.4319, 0.7471, 49.7467, 0.6828, 0.0005],
        [95, 49.4335, 0.7466, 49.7191, 0.6865, 0.0005],
        [96, 49.4395, 0.7422, 49.6586, 0.6661, 0.0005],
        [97, 49.4379, 0.7442, 49.7025, 0.6726, 0.0005],
        [98, 49.4276, 0.7587, 49.7083, 0.6718, 0.0005],
        [99, 49.4391, 0.7450, 49.7474, 0.7041, 0.0005],
        [100, 49.4290, 0.7590, 49.7056, 0.6935, 0.0005],
])


def main(results, label, color='black'):

    plt.figure("IoU", figsize=(6, 6))

    plt.plot(
        results[:, 0], results[:, 2], label='train ' + label, color=color)
    plt.plot(
        results[:, 0], results[:, 4], label='val ' + label, color=color, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()

    plt.figure("Loss", figsize=(6, 6))

    plt.plot(
        results[:, 0], results[:, 1], label='train ' + label, color=color)
    plt.plot(
        results[:, 0], results[:, 3], label='val ' + label, color=color, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    plt.ion()

    main(bce_results, 'BCE', color='b')
    main(class_balanced_bce_results, 'CB-BCE', color='r')

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()