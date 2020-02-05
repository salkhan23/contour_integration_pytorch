# ---------------------------------------------------------------------------------------
# Plot results of Model trained with Gaussian Regularization - different loss weights
#
# Model is baseline: n_iters=5, lateral kernel sizes = 15x15, gaussian width = 6
#
# Results_dir (gaussian_reg_weight_results): results/gaussian_reg_loss_weight_explore
# Results_dir (l1_reg_weight_results): results/l1_loss_weight_explore
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)

gaussian_reg_weight_results = {
    0: np.array([
        [0, 0.3170, 0.1868, 0.2554, 0.1972, 3e-05],
        [1, 0.2389, 0.2006, 0.2271, 0.1972, 3e-05],
        [2, 0.2134, 0.2008, 0.2098, 0.1987, 3e-05],
        [3, 0.1959, 0.2324, 0.1888, 0.2198, 3e-05],
        [4, 0.1842, 0.3186, 0.1795, 0.3470, 3e-05],
        [5, 0.1757, 0.3973, 0.1722, 0.4156, 3e-05],
        [6, 0.1690, 0.4698, 0.1663, 0.4966, 3e-05],
        [7, 0.1638, 0.5290, 0.1642, 0.5698, 3e-05],
        [8, 0.1600, 0.5750, 0.1593, 0.5918, 3e-05],
        [9, 0.1573, 0.6106, 0.1590, 0.6009, 3e-05],
        [10, 0.1552, 0.6404, 0.1562, 0.6318, 3e-05],
        [11, 0.1534, 0.6666, 0.1562, 0.6242, 3e-05],
        [12, 0.1521, 0.6880, 0.1542, 0.6563, 3e-05],
        [13, 0.1509, 0.7085, 0.1546, 0.6524, 3e-05],
        [14, 0.1500, 0.7241, 0.1542, 0.6650, 3e-05],
        [15, 0.1490, 0.7445, 0.1528, 0.6862, 3e-05],
        [16, 0.1481, 0.7613, 0.1532, 0.6792, 3e-05],
        [17, 0.1472, 0.7787, 0.1533, 0.6878, 3e-05],
        [18, 0.1464, 0.7959, 0.1533, 0.6868, 3e-05],
        [19, 0.1456, 0.8149, 0.1550, 0.6774, 3e-05],
        [20, 0.1448, 0.8339, 0.1534, 0.6822, 3e-05],
        [21, 0.1441, 0.8470, 0.1545, 0.6924, 3e-05],
        [22, 0.1434, 0.8643, 0.1545, 0.6699, 3e-05],
        [23, 0.1428, 0.8812, 0.1551, 0.6803, 3e-05],
        [24, 0.1421, 0.8979, 0.1559, 0.6885, 3e-05],
        [25, 0.1416, 0.9118, 0.1573, 0.6841, 3e-05],
        [26, 0.1410, 0.9256, 0.1567, 0.6931, 3e-05],
        [27, 0.1406, 0.9375, 0.1576, 0.6838, 3e-05],
        [28, 0.1402, 0.9465, 0.1589, 0.6800, 3e-05],
        [29, 0.1400, 0.9531, 0.1630, 0.6872, 3e-05],
        [30, 0.1398, 0.9577, 0.1618, 0.6746, 3e-05],
        [31, 0.1389, 0.9847, 0.1601, 0.7004, 3e-06],
        [32, 0.1388, 0.9875, 0.1605, 0.7051, 3e-06],
        [33, 0.1387, 0.9882, 0.1609, 0.7027, 3e-06],
        [34, 0.1387, 0.9888, 0.1608, 0.7056, 3e-06],
        [35, 0.1387, 0.9894, 0.1611, 0.7039, 3e-06],
        [36, 0.1386, 0.9903, 0.1614, 0.7041, 3e-06],
        [37, 0.1386, 0.9903, 0.1619, 0.7083, 3e-06],
        [38, 0.1386, 0.9910, 0.1619, 0.7049, 3e-06],
        [39, 0.1385, 0.9913, 0.1618, 0.7068, 3e-06],
        [40, 0.1385, 0.9913, 0.1623, 0.7037, 3e-06],
        [41, 0.1385, 0.9917, 0.1628, 0.7037, 3e-06],
        [42, 0.1385, 0.9918, 0.1626, 0.7070, 3e-06],
        [43, 0.1384, 0.9920, 0.1632, 0.7040, 3e-06],
        [44, 0.1384, 0.9921, 0.1634, 0.7051, 3e-06],
        [45, 0.1384, 0.9920, 0.1636, 0.7043, 3e-06],
        [46, 0.1384, 0.9923, 0.1633, 0.7099, 3e-06],
        [47, 0.1384, 0.9923, 0.1636, 0.7045, 3e-06],
        [48, 0.1384, 0.9924, 0.1645, 0.7036, 3e-06],
        [49, 0.1384, 0.9923, 0.1645, 0.7043, 3e-06],
    ]),

    0.01: np.array([
        [0, 2.6970, 0.1397, 0.4939, 0.1952, 3e-05],
        [1, 0.4666, 0.1988, 0.4360, 0.1972, 3e-05],
        [2, 0.4151, 0.1999, 0.3743, 0.1972, 3e-05],
        [3, 0.3712, 0.2001, 0.3482, 0.1972, 3e-05],
        [4, 0.3244, 0.2006, 0.2864, 0.1972, 3e-05],
        [5, 0.2909, 0.2006, 0.2692, 0.1972, 3e-05],
        [6, 0.2679, 0.2006, 0.2651, 0.1972, 3e-05],
        [7, 0.2531, 0.2006, 0.2420, 0.1972, 3e-05],
        [8, 0.2430, 0.2006, 0.2375, 0.1972, 3e-05],
        [9, 0.2369, 0.2006, 0.2388, 0.1972, 3e-05],
        [10, 0.2326, 0.2006, 0.2454, 0.1972, 3e-05],
        [11, 0.2301, 0.2006, 0.2315, 0.1972, 3e-05],
        [12, 0.2290, 0.2006, 0.2321, 0.1972, 3e-05],
        [13, 0.2279, 0.2006, 0.2278, 0.1972, 3e-05],
        [14, 0.2273, 0.2006, 0.2305, 0.1972, 3e-05],
        [15, 0.2266, 0.2006, 0.2280, 0.1972, 3e-05],
        [16, 0.2260, 0.2006, 0.2265, 0.1972, 3e-05],
        [17, 0.2255, 0.2006, 0.2260, 0.1973, 3e-05],
        [18, 0.2251, 0.2006, 0.2867, 0.1599, 3e-05],
        [19, 0.2248, 0.2006, 0.2261, 0.1973, 3e-05],
        [20, 0.2244, 0.2007, 0.2268, 0.1973, 3e-05],
        [21, 0.2241, 0.2009, 0.2252, 0.1980, 3e-05],
        [22, 0.2241, 0.2010, 0.2261, 0.1977, 3e-05],
        [23, 0.2239, 0.2010, 0.2269, 0.1983, 3e-05],
        [24, 0.2238, 0.2011, 0.2250, 0.1976, 3e-05],
        [25, 0.2236, 0.2014, 0.2288, 0.1973, 3e-05],
        [26, 0.2236, 0.2014, 0.2267, 0.1989, 3e-05],
        [27, 0.2234, 0.2018, 0.2247, 0.1979, 3e-05],
        [28, 0.2235, 0.2018, 0.2302, 0.1980, 3e-05],
        [29, 0.2235, 0.2017, 0.2368, 0.1896, 3e-05],
        [30, 0.2232, 0.2023, 0.2265, 0.1974, 3e-05],
        [31, 0.1994, 0.2027, 0.2005, 0.1987, 3e-06],
        [32, 0.1993, 0.2030, 0.2013, 0.1983, 3e-06],
        [33, 0.1992, 0.2030, 0.2004, 0.1990, 3e-06],
        [34, 0.1992, 0.2029, 0.2005, 0.1989, 3e-06],
        [35, 0.1992, 0.2029, 0.2006, 0.1991, 3e-06],
        [36, 0.1992, 0.2031, 0.2007, 0.1986, 3e-06],
        [37, 0.1992, 0.2033, 0.2018, 0.1983, 3e-06],
        [38, 0.1992, 0.2037, 0.2005, 0.1988, 3e-06],
        [39, 0.1992, 0.2033, 0.2005, 0.1992, 3e-06],
        [40, 0.1991, 0.2036, 0.2003, 0.1990, 3e-06],
        [41, 0.1991, 0.2038, 0.2007, 0.1983, 3e-06],
        [42, 0.1991, 0.2037, 0.2007, 0.1989, 3e-06],
        [43, 0.1991, 0.2039, 0.2003, 0.1995, 3e-06],
        [44, 0.1991, 0.2035, 0.2007, 0.1989, 3e-06],
        [45, 0.1991, 0.2038, 0.2003, 0.1994, 3e-06],
        [46, 0.1990, 0.2039, 0.2005, 0.1989, 3e-06],
        [47, 0.1991, 0.2038, 0.2004, 0.2009, 3e-06],
        [48, 0.1990, 0.2042, 0.2004, 0.1986, 3e-06],
        [49, 0.1990, 0.2037, 0.2003, 0.2005, 3e-06],
    ]),

    0.0001: np.array([
        [0, 0.4175, 0.1849, 0.2922, 0.1992, 3e-05],
        [1, 0.2697, 0.1996, 0.2471, 0.1992, 3e-05],
        [2, 0.2342, 0.1996, 0.2225, 0.1992, 3e-05],
        [3, 0.2166, 0.1996, 0.2068, 0.1995, 3e-05],
        [4, 0.2042, 0.2027, 0.1994, 0.2010, 3e-05],
        [5, 0.1970, 0.2488, 0.1970, 0.3092, 3e-05],
        [6, 0.1925, 0.2962, 0.1966, 0.2249, 3e-05],
        [7, 0.1895, 0.3183, 0.1878, 0.2879, 3e-05],
        [8, 0.1865, 0.3458, 0.1841, 0.3879, 3e-05],
        [9, 0.1837, 0.3796, 0.1834, 0.3886, 3e-05],
        [10, 0.1810, 0.4138, 0.1836, 0.3500, 3e-05],
        [11, 0.1785, 0.4489, 0.1770, 0.4929, 3e-05],
        [12, 0.1764, 0.4775, 0.1760, 0.4925, 3e-05],
        [13, 0.1746, 0.5040, 0.1757, 0.4855, 3e-05],
        [14, 0.1733, 0.5207, 0.1736, 0.5167, 3e-05],
        [15, 0.1718, 0.5431, 0.1732, 0.5029, 3e-05],
        [16, 0.1708, 0.5548, 0.1716, 0.5673, 3e-05],
        [17, 0.1698, 0.5708, 0.1699, 0.5591, 3e-05],
        [18, 0.1692, 0.5780, 0.1865, 0.4506, 3e-05],
        [19, 0.1684, 0.5904, 0.1688, 0.5874, 3e-05],
        [20, 0.1676, 0.5975, 0.1723, 0.5767, 3e-05],
        [21, 0.1671, 0.6066, 0.1677, 0.5842, 3e-05],
        [22, 0.1664, 0.6142, 0.1678, 0.5944, 3e-05],
        [23, 0.1659, 0.6199, 0.1668, 0.6147, 3e-05],
        [24, 0.1655, 0.6264, 0.1656, 0.6308, 3e-05],
        [25, 0.1651, 0.6301, 0.1660, 0.6306, 3e-05],
        [26, 0.1647, 0.6367, 0.1666, 0.6207, 3e-05],
        [27, 0.1643, 0.6429, 0.1653, 0.6310, 3e-05],
        [28, 0.1640, 0.6466, 0.1659, 0.6305, 3e-05],
        [29, 0.1636, 0.6522, 0.1652, 0.6427, 3e-05],
        [30, 0.1633, 0.6570, 0.1660, 0.6363, 3e-05],
        [31, 0.1599, 0.6876, 0.1612, 0.6605, 3e-06],
        [32, 0.1594, 0.6886, 0.1612, 0.6599, 3e-06],
        [33, 0.1593, 0.6901, 0.1616, 0.6510, 3e-06],
        [34, 0.1592, 0.6919, 0.1610, 0.6625, 3e-06],
        [35, 0.1591, 0.6928, 0.1611, 0.6639, 3e-06],
        [36, 0.1590, 0.6924, 0.1608, 0.6660, 3e-06],
        [37, 0.1590, 0.6950, 0.1609, 0.6674, 3e-06],
        [38, 0.1589, 0.6932, 0.1608, 0.6658, 3e-06],
        [39, 0.1588, 0.6947, 0.1609, 0.6592, 3e-06],
        [40, 0.1588, 0.6957, 0.1608, 0.6601, 3e-06],
        [41, 0.1587, 0.6963, 0.1608, 0.6595, 3e-06],
        [42, 0.1587, 0.6968, 0.1610, 0.6677, 3e-06],
        [43, 0.1587, 0.6951, 0.1606, 0.6684, 3e-06],
        [44, 0.1586, 0.6975, 0.1608, 0.6624, 3e-06],
        [45, 0.1585, 0.6978, 0.1606, 0.6674, 3e-06],
        [46, 0.1585, 0.7001, 0.1606, 0.6665, 3e-06],
        [47, 0.1585, 0.6994, 0.1606, 0.6645, 3e-06],
        [48, 0.1584, 0.7002, 0.1611, 0.6563, 3e-06],
        [49, 0.1584, 0.7004, 0.1616, 0.6527, 3e-06],
    ]),

    1e-5: np.array([
        [0, 0.2930, 0.1932, 0.2439, 0.2027, 3e-05],
        [1, 0.2241, 0.1995, 0.2211, 0.2027, 3e-05],
        [2, 0.2046, 0.2009, 0.1993, 0.2041, 3e-05],
        [3, 0.1936, 0.2421, 0.1892, 0.2924, 3e-05],
        [4, 0.1857, 0.3550, 0.1876, 0.4110, 3e-05],
        [5, 0.1795, 0.4224, 0.1773, 0.4321, 3e-05],
        [6, 0.1745, 0.4821, 0.1741, 0.5049, 3e-05],
        [7, 0.1707, 0.5294, 0.1807, 0.5067, 3e-05],
        [8, 0.1677, 0.5680, 0.1681, 0.5737, 3e-05],
        [9, 0.1655, 0.5986, 0.1664, 0.6073, 3e-05],
        [10, 0.1638, 0.6227, 0.1646, 0.6179, 3e-05],
        [11, 0.1622, 0.6451, 0.1631, 0.6353, 3e-05],
        [12, 0.1611, 0.6613, 0.1631, 0.6295, 3e-05],
        [13, 0.1599, 0.6776, 0.1615, 0.6586, 3e-05],
        [14, 0.1590, 0.6925, 0.1610, 0.6766, 3e-05],
        [15, 0.1581, 0.7057, 0.1664, 0.5851, 3e-05],
        [16, 0.1574, 0.7180, 0.1604, 0.6663, 3e-05],
        [17, 0.1568, 0.7249, 0.1609, 0.6621, 3e-05],
        [18, 0.1561, 0.7372, 0.1595, 0.6850, 3e-05],
        [19, 0.1555, 0.7482, 0.1593, 0.6945, 3e-05],
        [20, 0.1550, 0.7572, 0.1601, 0.6711, 3e-05],
        [21, 0.1545, 0.7670, 0.1602, 0.6687, 3e-05],
        [22, 0.1540, 0.7743, 0.1598, 0.6912, 3e-05],
        [23, 0.1536, 0.7840, 0.1593, 0.6782, 3e-05],
        [24, 0.1531, 0.7927, 0.1582, 0.7094, 3e-05],
        [25, 0.1527, 0.8005, 0.1588, 0.7028, 3e-05],
        [26, 0.1524, 0.8054, 0.1585, 0.7085, 3e-05],
        [27, 0.1521, 0.8145, 0.1583, 0.7100, 3e-05],
        [28, 0.1518, 0.8222, 0.1581, 0.7145, 3e-05],
        [29, 0.1514, 0.8292, 0.1592, 0.7100, 3e-05],
        [30, 0.1511, 0.8371, 0.1594, 0.6993, 3e-05],
        [31, 0.1488, 0.8966, 0.1574, 0.7289, 3e-06],
        [32, 0.1484, 0.9045, 0.1574, 0.7264, 3e-06],
        [33, 0.1482, 0.9084, 0.1577, 0.7289, 3e-06],
        [34, 0.1480, 0.9109, 0.1575, 0.7302, 3e-06],
        [35, 0.1479, 0.9133, 0.1576, 0.7310, 3e-06],
        [36, 0.1478, 0.9156, 0.1576, 0.7318, 3e-06],
        [37, 0.1477, 0.9167, 0.1574, 0.7253, 3e-06],
        [38, 0.1476, 0.9179, 0.1576, 0.7279, 3e-06],
        [39, 0.1475, 0.9188, 0.1575, 0.7258, 3e-06],
        [40, 0.1475, 0.9207, 0.1582, 0.7285, 3e-06],
        [41, 0.1474, 0.9222, 0.1578, 0.7277, 3e-06],
        [42, 0.1473, 0.9237, 0.1579, 0.7278, 3e-06],
        [43, 0.1472, 0.9253, 0.1579, 0.7251, 3e-06],
        [44, 0.1472, 0.9275, 0.1580, 0.7289, 3e-06],
        [45, 0.1471, 0.9275, 0.1584, 0.7279, 3e-06],
        [46, 0.1470, 0.9299, 0.1581, 0.7187, 3e-06],
        [47, 0.1470, 0.9311, 0.1580, 0.7279, 3e-06],
        [48, 0.1469, 0.9329, 0.1583, 0.7277, 3e-06],
        [49, 0.1468, 0.9333, 0.1586, 0.7311, 3e-06]
    ]),

    1e-6: np.array([
        [0, 0.3366, 0.1872, 0.2606, 0.1907, 3e-05],
        [1, 0.2422, 0.2019, 0.2201, 0.1907, 3e-05],
        [2, 0.2123, 0.2058, 0.2042, 0.2186, 3e-05],
        [3, 0.1942, 0.2571, 0.1968, 0.3372, 3e-05],
        [4, 0.1830, 0.3337, 0.1785, 0.3414, 3e-05],
        [5, 0.1749, 0.4194, 0.1726, 0.4568, 3e-05],
        [6, 0.1687, 0.4852, 0.1672, 0.5186, 3e-05],
        [7, 0.1643, 0.5380, 0.1633, 0.5454, 3e-05],
        [8, 0.1611, 0.5789, 0.1628, 0.5846, 3e-05],
        [9, 0.1588, 0.6134, 0.1606, 0.5747, 3e-05],
        [10, 0.1570, 0.6383, 0.1588, 0.6035, 3e-05],
        [11, 0.1555, 0.6621, 0.1572, 0.6382, 3e-05],
        [12, 0.1544, 0.6804, 0.1569, 0.6199, 3e-05],
        [13, 0.1533, 0.7022, 0.1571, 0.6368, 3e-05],
        [14, 0.1523, 0.7186, 0.1561, 0.6580, 3e-05],
        [15, 0.1515, 0.7322, 0.1555, 0.6668, 3e-05],
        [16, 0.1507, 0.7498, 0.1553, 0.6720, 3e-05],
        [17, 0.1500, 0.7629, 0.1562, 0.6653, 3e-05],
        [18, 0.1493, 0.7789, 0.1549, 0.6773, 3e-05],
        [19, 0.1485, 0.7968, 0.1560, 0.6797, 3e-05],
        [20, 0.1479, 0.8092, 0.1556, 0.6733, 3e-05],
        [21, 0.1473, 0.8254, 0.1551, 0.6947, 3e-05],
        [22, 0.1466, 0.8410, 0.1557, 0.6766, 3e-05],
        [23, 0.1460, 0.8540, 0.1562, 0.6886, 3e-05],
        [24, 0.1454, 0.8698, 0.1567, 0.6789, 3e-05],
        [25, 0.1449, 0.8827, 0.1569, 0.6831, 3e-05],
        [26, 0.1444, 0.8990, 0.1583, 0.6839, 3e-05],
        [27, 0.1439, 0.9071, 0.1578, 0.6831, 3e-05],
        [28, 0.1435, 0.9206, 0.1595, 0.6867, 3e-05],
        [29, 0.1431, 0.9342, 0.1611, 0.6844, 3e-05],
        [30, 0.1429, 0.9378, 0.1609, 0.6853, 3e-05],
        [31, 0.1417, 0.9770, 0.1599, 0.7001, 3e-06],
        [32, 0.1415, 0.9822, 0.1598, 0.6991, 3e-06],
        [33, 0.1414, 0.9847, 0.1601, 0.6991, 3e-06],
        [34, 0.1413, 0.9862, 0.1605, 0.6995, 3e-06],
        [35, 0.1412, 0.9863, 0.1608, 0.6961, 3e-06],
        [36, 0.1412, 0.9874, 0.1611, 0.6980, 3e-06],
        [37, 0.1411, 0.9879, 0.1608, 0.7021, 3e-06],
        [38, 0.1411, 0.9886, 0.1613, 0.6997, 3e-06],
        [39, 0.1410, 0.9892, 0.1614, 0.6988, 3e-06],
        [40, 0.1410, 0.9895, 0.1615, 0.6984, 3e-06],
        [41, 0.1410, 0.9900, 0.1617, 0.7014, 3e-06],
        [42, 0.1409, 0.9902, 0.1622, 0.7022, 3e-06],
        [43, 0.1409, 0.9907, 0.1625, 0.6992, 3e-06],
        [44, 0.1408, 0.9911, 0.1624, 0.6974, 3e-06],
        [45, 0.1408, 0.9912, 0.1627, 0.6992, 3e-06],
        [46, 0.1408, 0.9914, 0.1629, 0.7013, 3e-06],
        [47, 0.1407, 0.9913, 0.1628, 0.6999, 3e-06],
        [48, 0.1407, 0.9914, 0.1631, 0.6997, 3e-06],
        [49, 0.1407, 0.9917, 0.1637, 0.7012, 3e-06],
    ])
}
l1_reg_weight_results = {
    0: np.array([
        [0, 0.5456, 0.1653, 0.4974, 0.1972, 3e-05],
        [1, 0.4587, 0.2006, 0.4221, 0.1972, 3e-05],
        [2, 0.4002, 0.2006, 0.3797, 0.1972, 3e-05],
        [3, 0.3498, 0.2006, 0.3250, 0.1972, 3e-05],
        [4, 0.3073, 0.2006, 0.2861, 0.1972, 3e-05],
        [5, 0.2723, 0.2009, 0.2517, 0.1983, 3e-05],
        [6, 0.2444, 0.2352, 0.2354, 0.2711, 3e-05],
        [7, 0.2227, 0.3057, 0.2132, 0.3361, 3e-05],
        [8, 0.2052, 0.3785, 0.2027, 0.4167, 3e-05],
        [9, 0.1914, 0.4484, 0.1906, 0.4715, 3e-05],
        [10, 0.1808, 0.4998, 0.1745, 0.4980, 3e-05],
        [11, 0.1728, 0.5411, 0.1705, 0.4936, 3e-05],
        [12, 0.1668, 0.5761, 0.1669, 0.5689, 3e-05],
        [13, 0.1624, 0.6034, 0.1613, 0.6021, 3e-05],
        [14, 0.1590, 0.6296, 0.1597, 0.6098, 3e-05],
        [15, 0.1565, 0.6522, 0.1573, 0.6419, 3e-05],
        [16, 0.1545, 0.6715, 0.1563, 0.6453, 3e-05],
        [17, 0.1530, 0.6875, 0.1556, 0.6503, 3e-05],
        [18, 0.1518, 0.7054, 0.1545, 0.6606, 3e-05],
        [19, 0.1507, 0.7222, 0.1547, 0.6539, 3e-05],
        [20, 0.1497, 0.7346, 0.1536, 0.6654, 3e-05],
        [21, 0.1488, 0.7507, 0.1536, 0.6785, 3e-05],
        [22, 0.1480, 0.7675, 0.1532, 0.6841, 3e-05],
        [23, 0.1472, 0.7825, 0.1540, 0.6649, 3e-05],
        [24, 0.1464, 0.8014, 0.1554, 0.6584, 3e-05],
        [25, 0.1456, 0.8195, 0.1538, 0.6871, 3e-05],
        [26, 0.1451, 0.8282, 0.1541, 0.6934, 3e-05],
        [27, 0.1442, 0.8489, 0.1541, 0.6853, 3e-05],
        [28, 0.1436, 0.8641, 0.1544, 0.6839, 3e-05],
        [29, 0.1429, 0.8806, 0.1549, 0.6761, 3e-05],
        [30, 0.1423, 0.8939, 0.1557, 0.6845, 3e-05],
        [31, 0.1406, 0.9485, 0.1553, 0.7033, 3e-06],
        [32, 0.1404, 0.9550, 0.1551, 0.7003, 3e-06],
        [33, 0.1403, 0.9585, 0.1555, 0.6957, 3e-06],
        [34, 0.1402, 0.9616, 0.1556, 0.6955, 3e-06],
        [35, 0.1401, 0.9643, 0.1559, 0.6961, 3e-06],
        [36, 0.1400, 0.9667, 0.1560, 0.6964, 3e-06],
        [37, 0.1399, 0.9681, 0.1560, 0.6928, 3e-06],
        [38, 0.1398, 0.9703, 0.1563, 0.6974, 3e-06],
        [39, 0.1398, 0.9725, 0.1562, 0.6949, 3e-06],
        [40, 0.1397, 0.9734, 0.1567, 0.6941, 3e-06],
        [41, 0.1396, 0.9751, 0.1571, 0.6957, 3e-06],
        [42, 0.1396, 0.9767, 0.1568, 0.6933, 3e-06],
        [43, 0.1395, 0.9779, 0.1572, 0.6936, 3e-06],
        [44, 0.1395, 0.9789, 0.1570, 0.6937, 3e-06],
        [45, 0.1394, 0.9797, 0.1572, 0.6922, 3e-06],
        [46, 0.1393, 0.9812, 0.1573, 0.6923, 3e-06],
        [47, 0.1393, 0.9819, 0.1575, 0.6947, 3e-06],
        [48, 0.1392, 0.9831, 0.1578, 0.6913, 3e-06],
        [49, 0.1392, 0.9838, 0.1580, 0.6935, 3e-06],
    ]),

    0.01: np.array([
        [0, 6.4812, 0.0194, 0.6934, 0.0196, 3e-05],
        [1, 0.6696, 0.0173, 0.6459, 0.0184, 3e-05],
        [2, 0.6250, 0.0438, 0.6001, 0.0866, 3e-05],
        [3, 0.5807, 0.1132, 0.5522, 0.1717, 3e-05],
        [4, 0.5312, 0.1611, 0.5009, 0.1892, 3e-05],
        [5, 0.4849, 0.1820, 0.4596, 0.1951, 3e-05],
        [6, 0.4404, 0.1927, 0.4214, 0.1965, 3e-05],
        [7, 0.3982, 0.1978, 0.3744, 0.1973, 3e-05],
        [8, 0.3593, 0.1992, 0.3328, 0.1972, 3e-05],
        [9, 0.3276, 0.2004, 0.3084, 0.1972, 3e-05],
        [10, 0.3045, 0.2003, 0.2919, 0.1972, 3e-05],
        [11, 0.2873, 0.2005, 0.2833, 0.1972, 3e-05],
        [12, 0.2759, 0.2005, 0.2704, 0.1972, 3e-05],
        [13, 0.2690, 0.2006, 0.2662, 0.1972, 3e-05],
        [14, 0.2651, 0.2006, 0.2644, 0.1972, 3e-05],
        [15, 0.2633, 0.2006, 0.2636, 0.1972, 3e-05],
        [16, 0.2625, 0.2006, 0.2640, 0.1972, 3e-05],
        [17, 0.2623, 0.2006, 0.2639, 0.1972, 3e-05],
        [18, 0.2621, 0.2006, 0.2636, 0.1972, 3e-05],
        [19, 0.2619, 0.2005, 0.2634, 0.1972, 3e-05],
        [20, 0.2618, 0.2006, 0.2626, 0.1972, 3e-05],
        [21, 0.2616, 0.2008, 0.2623, 0.1973, 3e-05],
        [22, 0.2614, 0.2009, 0.2625, 0.1973, 3e-05],
        [23, 0.2614, 0.2009, 0.2617, 0.1973, 3e-05],
        [24, 0.2614, 0.2011, 0.2615, 0.1974, 3e-05],
        [25, 0.2611, 0.2011, 0.2634, 0.1984, 3e-05],
        [26, 0.2610, 0.2014, 0.2624, 0.1980, 3e-05],
        [27, 0.2610, 0.2014, 0.2612, 0.1981, 3e-05],
        [28, 0.2610, 0.2015, 0.2618, 0.1987, 3e-05],
        [29, 0.2607, 0.2018, 0.2621, 0.1990, 3e-05],
        [30, 0.2607, 0.2020, 0.2614, 0.1995, 3e-05],
        [31, 0.2042, 0.2019, 0.2048, 0.1996, 3e-06],
        [32, 0.2036, 0.2021, 0.2045, 0.1993, 3e-06],
        [33, 0.2036, 0.2020, 0.2046, 0.1996, 3e-06],
        [34, 0.2036, 0.2021, 0.2046, 0.1994, 3e-06],
        [35, 0.2036, 0.2021, 0.2046, 0.1996, 3e-06],
        [36, 0.2036, 0.2021, 0.2046, 0.1993, 3e-06],
        [37, 0.2035, 0.2022, 0.2046, 0.1995, 3e-06],
        [38, 0.2035, 0.2021, 0.2046, 0.1995, 3e-06],
        [39, 0.2035, 0.2023, 0.2046, 0.1995, 3e-06],
        [40, 0.2035, 0.2024, 0.2046, 0.1996, 3e-06],
        [41, 0.2035, 0.2022, 0.2046, 0.1999, 3e-06],
        [42, 0.2035, 0.2023, 0.2045, 0.1997, 3e-06],
        [43, 0.2035, 0.2024, 0.2046, 0.1997, 3e-06],
        [44, 0.2035, 0.2023, 0.2045, 0.1995, 3e-06],
        [45, 0.2035, 0.2024, 0.2047, 0.1995, 3e-06],
        [46, 0.2035, 0.2023, 0.2045, 0.1996, 3e-06],
        [47, 0.2035, 0.2022, 0.2045, 0.1993, 3e-06],
        [48, 0.2035, 0.2023, 0.2045, 0.1998, 3e-06],
        [49, 0.2035, 0.2023, 0.2045, 0.1999, 3e-06],
    ]),

    0.001: np.array([
        [0, 1.1816, 0.0271, 0.5255, 0.0378, 3e-05],
        [1, 0.4812, 0.0861, 0.4091, 0.1990, 3e-05],
        [2, 0.3661, 0.1987, 0.3139, 0.1987, 3e-05],
        [3, 0.3078, 0.1992, 0.3015, 0.1992, 3e-05],
        [4, 0.2752, 0.1996, 0.2574, 0.1992, 3e-05],
        [5, 0.2548, 0.1996, 0.2532, 0.1992, 3e-05],
        [6, 0.2419, 0.1996, 0.2356, 0.1992, 3e-05],
        [7, 0.2344, 0.1996, 0.2299, 0.1992, 3e-05],
        [8, 0.2295, 0.1996, 0.2277, 0.1992, 3e-05],
        [9, 0.2265, 0.1996, 0.2252, 0.1992, 3e-05],
        [10, 0.2241, 0.1996, 0.2231, 0.1992, 3e-05],
        [11, 0.2222, 0.1996, 0.2205, 0.1992, 3e-05],
        [12, 0.2204, 0.1995, 0.2199, 0.1992, 3e-05],
        [13, 0.2188, 0.1993, 0.2168, 0.1993, 3e-05],
        [14, 0.2174, 0.1996, 0.2162, 0.1992, 3e-05],
        [15, 0.2163, 0.1996, 0.2163, 0.1993, 3e-05],
        [16, 0.2157, 0.1996, 0.2155, 0.1992, 3e-05],
        [17, 0.2146, 0.1996, 0.2138, 0.1992, 3e-05],
        [18, 0.2142, 0.1997, 0.2124, 0.1992, 3e-05],
        [19, 0.2131, 0.1996, 0.2129, 0.1992, 3e-05],
        [20, 0.2124, 0.1996, 0.2103, 0.1992, 3e-05],
        [21, 0.2117, 0.1997, 0.2115, 0.1994, 3e-05],
        [22, 0.2118, 0.1998, 0.2127, 0.1993, 3e-05],
        [23, 0.2108, 0.1997, 0.2103, 0.1992, 3e-05],
        [24, 0.2107, 0.1999, 0.2102, 0.1992, 3e-05],
        [25, 0.2103, 0.1999, 0.2109, 0.1993, 3e-05],
        [26, 0.2097, 0.1998, 0.2126, 0.1999, 3e-05],
        [27, 0.2095, 0.2001, 0.2101, 0.1993, 3e-05],
        [28, 0.2094, 0.2002, 0.2116, 0.2004, 3e-05],
        [29, 0.2086, 0.2002, 0.2093, 0.1994, 3e-05],
        [30, 0.2100, 0.2000, 0.2080, 0.1993, 3e-05],
        [31, 0.1993, 0.2003, 0.1990, 0.1997, 3e-06],
        [32, 0.1992, 0.2004, 0.1990, 0.2001, 3e-06],
        [33, 0.1991, 0.2005, 0.1991, 0.2003, 3e-06],
        [34, 0.1991, 0.2004, 0.1990, 0.2000, 3e-06],
        [35, 0.1991, 0.2004, 0.1991, 0.2000, 3e-06],
        [36, 0.1990, 0.2004, 0.1989, 0.2000, 3e-06],
        [37, 0.1990, 0.2005, 0.1990, 0.2001, 3e-06],
        [38, 0.1990, 0.2003, 0.1988, 0.2000, 3e-06],
        [39, 0.1990, 0.2004, 0.1988, 0.2000, 3e-06],
        [40, 0.1990, 0.2005, 0.1988, 0.2001, 3e-06],
        [41, 0.1990, 0.2003, 0.1991, 0.2001, 3e-06],
        [42, 0.1989, 0.2004, 0.1987, 0.1998, 3e-06],
        [43, 0.1989, 0.2003, 0.1988, 0.2004, 3e-06],
        [44, 0.1988, 0.2005, 0.1987, 0.2001, 3e-06],
        [45, 0.1988, 0.2006, 0.1986, 0.1999, 3e-06],
        [46, 0.1988, 0.2005, 0.1988, 0.1999, 3e-06],
        [47, 0.1988, 0.2003, 0.1988, 0.1997, 3e-06],
        [48, 0.1987, 0.2004, 0.1987, 0.2007, 3e-06],
        [49, 0.1987, 0.2005, 0.1985, 0.1999, 3e-06],
    ]),

    0.0001: np.array([
        [0, 0.5560, 0.1675, 0.3952, 0.2022, 3e-05],
        [1, 0.3499, 0.1992, 0.3194, 0.2027, 3e-05],
        [2, 0.2908, 0.1994, 0.2786, 0.2027, 3e-05],
        [3, 0.2552, 0.1994, 0.2434, 0.2027, 3e-05],
        [4, 0.2339, 0.1994, 0.2244, 0.2027, 3e-05],
        [5, 0.2218, 0.1994, 0.2199, 0.2027, 3e-05],
        [6, 0.2152, 0.1994, 0.2115, 0.2027, 3e-05],
        [7, 0.2117, 0.1994, 0.2106, 0.2027, 3e-05],
        [8, 0.2096, 0.1994, 0.2086, 0.2027, 3e-05],
        [9, 0.2083, 0.1994, 0.2103, 0.2027, 3e-05],
        [10, 0.2072, 0.1994, 0.2062, 0.2027, 3e-05],
        [11, 0.2061, 0.1996, 0.2048, 0.2028, 3e-05],
        [12, 0.2048, 0.2005, 0.2049, 0.2028, 3e-05],
        [13, 0.2003, 0.2193, 0.1989, 0.2164, 3e-05],
        [14, 0.1968, 0.2442, 0.1960, 0.2565, 3e-05],
        [15, 0.1952, 0.2567, 0.1963, 0.2500, 3e-05],
        [16, 0.1940, 0.2637, 0.1944, 0.2655, 3e-05],
        [17, 0.1925, 0.2727, 0.2015, 0.3174, 3e-05],
        [18, 0.1913, 0.2782, 0.1924, 0.2601, 3e-05],
        [19, 0.1908, 0.2832, 0.1951, 0.2403, 3e-05],
        [20, 0.1900, 0.2862, 0.1909, 0.2664, 3e-05],
        [21, 0.1892, 0.2900, 0.1907, 0.2762, 3e-05],
        [22, 0.1887, 0.2945, 0.1909, 0.2666, 3e-05],
        [23, 0.1882, 0.2982, 0.1893, 0.3126, 3e-05],
        [24, 0.1880, 0.2962, 0.1887, 0.2827, 3e-05],
        [25, 0.1870, 0.3082, 0.1882, 0.3170, 3e-05],
        [26, 0.1868, 0.3106, 0.1888, 0.2749, 3e-05],
        [27, 0.1863, 0.3146, 0.1870, 0.2979, 3e-05],
        [28, 0.1860, 0.3183, 0.1935, 0.3448, 3e-05],
        [29, 0.1862, 0.3183, 0.1876, 0.2781, 3e-05],
        [30, 0.1853, 0.3245, 0.1876, 0.2914, 3e-05],
        [31, 0.1818, 0.3341, 0.1837, 0.3124, 3e-06],
        [32, 0.1812, 0.3404, 0.1827, 0.3344, 3e-06],
        [33, 0.1811, 0.3425, 0.1841, 0.3111, 3e-06],
        [34, 0.1810, 0.3440, 0.1830, 0.3311, 3e-06],
        [35, 0.1809, 0.3438, 0.1826, 0.3369, 3e-06],
        [36, 0.1809, 0.3457, 0.1824, 0.3410, 3e-06],
        [37, 0.1808, 0.3461, 0.1828, 0.3482, 3e-06],
        [38, 0.1807, 0.3456, 0.1824, 0.3344, 3e-06],
        [39, 0.1807, 0.3470, 0.1824, 0.3534, 3e-06],
        [40, 0.1807, 0.3476, 0.1828, 0.3225, 3e-06],
        [41, 0.1806, 0.3479, 0.1824, 0.3326, 3e-06],
        [42, 0.1806, 0.3481, 0.1822, 0.3417, 3e-06],
        [43, 0.1805, 0.3476, 0.1826, 0.3273, 3e-06],
        [44, 0.1804, 0.3494, 0.1821, 0.3475, 3e-06],
        [45, 0.1804, 0.3488, 0.1824, 0.3374, 3e-06],
        [46, 0.1804, 0.3500, 0.1820, 0.3473, 3e-06],
        [47, 0.1803, 0.3495, 0.1824, 0.3307, 3e-06],
        [48, 0.1802, 0.3493, 0.1831, 0.3142, 3e-06],
        [49, 0.1802, 0.3495, 0.1824, 0.3679, 3e-06],
    ]),

    1e-5: np.array([
        [0, 0.5474, 0.1692, 0.4754, 0.1907, 3e-05],
        [1, 0.4264, 0.2019, 0.3891, 0.1907, 3e-05],
        [2, 0.3561, 0.2019, 0.3350, 0.1907, 3e-05],
        [3, 0.3044, 0.2019, 0.2869, 0.1907, 3e-05],
        [4, 0.2673, 0.2019, 0.2559, 0.1907, 3e-05],
        [5, 0.2416, 0.2019, 0.2295, 0.1907, 3e-05],
        [6, 0.2242, 0.2019, 0.2181, 0.1907, 3e-05],
        [7, 0.2109, 0.2023, 0.2070, 0.1940, 3e-05],
        [8, 0.2009, 0.2208, 0.1951, 0.2213, 3e-05],
        [9, 0.1944, 0.2519, 0.1904, 0.2306, 3e-05],
        [10, 0.1893, 0.2804, 0.1878, 0.2882, 3e-05],
        [11, 0.1857, 0.3087, 0.1839, 0.3078, 3e-05],
        [12, 0.1825, 0.3441, 0.1821, 0.3642, 3e-05],
        [13, 0.1801, 0.3753, 0.1792, 0.3625, 3e-05],
        [14, 0.1779, 0.4087, 0.1782, 0.3803, 3e-05],
        [15, 0.1761, 0.4346, 0.1782, 0.4343, 3e-05],
        [16, 0.1745, 0.4574, 0.1756, 0.4395, 3e-05],
        [17, 0.1732, 0.4766, 0.1750, 0.4296, 3e-05],
        [18, 0.1720, 0.4954, 0.1725, 0.4805, 3e-05],
        [19, 0.1709, 0.5099, 0.1716, 0.5058, 3e-05],
        [20, 0.1698, 0.5247, 0.1714, 0.5005, 3e-05],
        [21, 0.1688, 0.5386, 0.1707, 0.5116, 3e-05],
        [22, 0.1682, 0.5501, 0.1699, 0.5329, 3e-05],
        [23, 0.1673, 0.5628, 0.1691, 0.5322, 3e-05],
        [24, 0.1665, 0.5739, 0.1689, 0.5399, 3e-05],
        [25, 0.1659, 0.5833, 0.1683, 0.5225, 3e-05],
        [26, 0.1653, 0.5924, 0.1675, 0.5668, 3e-05],
        [27, 0.1647, 0.6038, 0.1670, 0.5624, 3e-05],
        [28, 0.1642, 0.6125, 0.1711, 0.5195, 3e-05],
        [29, 0.1637, 0.6192, 0.1673, 0.5813, 3e-05],
        [30, 0.1633, 0.6294, 0.1666, 0.5835, 3e-05],
        [31, 0.1605, 0.6665, 0.1642, 0.6037, 3e-06],
        [32, 0.1600, 0.6725, 0.1640, 0.6089, 3e-06],
        [33, 0.1597, 0.6765, 0.1640, 0.6071, 3e-06],
        [34, 0.1596, 0.6767, 0.1641, 0.6037, 3e-06],
        [35, 0.1595, 0.6789, 0.1640, 0.6088, 3e-06],
        [36, 0.1594, 0.6808, 0.1639, 0.6061, 3e-06],
        [37, 0.1593, 0.6826, 0.1642, 0.6018, 3e-06],
        [38, 0.1592, 0.6839, 0.1642, 0.6053, 3e-06],
        [39, 0.1591, 0.6853, 0.1640, 0.6051, 3e-06],
        [40, 0.1590, 0.6857, 0.1636, 0.6149, 3e-06],
        [41, 0.1589, 0.6866, 0.1637, 0.6093, 3e-06],
        [42, 0.1589, 0.6895, 0.1636, 0.6099, 3e-06],
        [43, 0.1588, 0.6898, 0.1635, 0.6116, 3e-06],
        [44, 0.1587, 0.6908, 0.1637, 0.6078, 3e-06],
        [45, 0.1587, 0.6918, 0.1640, 0.6101, 3e-06],
        [46, 0.1585, 0.6939, 0.1634, 0.6154, 3e-06],
        [47, 0.1585, 0.6944, 0.1635, 0.6118, 3e-06],
        [48, 0.1584, 0.6968, 0.1635, 0.6170, 3e-06],
        [49, 0.1583, 0.6968, 0.1635, 0.6156, 3e-06],
    ]),

    1e-6: np.array([
        [0, 0.4271, 0.1723, 0.3494, 0.1947, 3e-05],
        [1, 0.3263, 0.1961, 0.3065, 0.1947, 3e-05],
        [2, 0.2762, 0.1961, 0.2441, 0.1947, 3e-05],
        [3, 0.2415, 0.1961, 0.2326, 0.1949, 3e-05],
        [4, 0.2172, 0.2039, 0.2059, 0.2066, 3e-05],
        [5, 0.2006, 0.2860, 0.1932, 0.3100, 3e-05],
        [6, 0.1887, 0.3603, 0.1839, 0.3225, 3e-05],
        [7, 0.1798, 0.4278, 0.1773, 0.4575, 3e-05],
        [8, 0.1732, 0.4870, 0.1708, 0.5055, 3e-05],
        [9, 0.1682, 0.5348, 0.1665, 0.5274, 3e-05],
        [10, 0.1647, 0.5708, 0.1639, 0.5618, 3e-05],
        [11, 0.1620, 0.6016, 0.1621, 0.5939, 3e-05],
        [12, 0.1602, 0.6273, 0.1611, 0.6228, 3e-05],
        [13, 0.1588, 0.6488, 0.1594, 0.6346, 3e-05],
        [14, 0.1574, 0.6696, 0.1593, 0.6357, 3e-05],
        [15, 0.1565, 0.6850, 0.1581, 0.6596, 3e-05],
        [16, 0.1556, 0.6976, 0.1590, 0.6554, 3e-05],
        [17, 0.1548, 0.7146, 0.1584, 0.6517, 3e-05],
        [18, 0.1540, 0.7291, 0.1579, 0.6538, 3e-05],
        [19, 0.1534, 0.7400, 0.1575, 0.6740, 3e-05],
        [20, 0.1527, 0.7548, 0.1573, 0.6794, 3e-05],
        [21, 0.1521, 0.7670, 0.1571, 0.6797, 3e-05],
        [22, 0.1515, 0.7794, 0.1571, 0.6851, 3e-05],
        [23, 0.1509, 0.7926, 0.1586, 0.6768, 3e-05],
        [24, 0.1505, 0.8030, 0.1573, 0.6938, 3e-05],
        [25, 0.1498, 0.8185, 0.1577, 0.6938, 3e-05],
        [26, 0.1494, 0.8287, 0.1594, 0.6786, 3e-05],
        [27, 0.1488, 0.8420, 0.1580, 0.6975, 3e-05],
        [28, 0.1484, 0.8531, 0.1579, 0.6885, 3e-05],
        [29, 0.1479, 0.8654, 0.1590, 0.7039, 3e-05],
        [30, 0.1475, 0.8758, 0.1596, 0.6951, 3e-05],
        [31, 0.1456, 0.9343, 0.1583, 0.7107, 3e-06],
        [32, 0.1453, 0.9438, 0.1588, 0.7112, 3e-06],
        [33, 0.1452, 0.9469, 0.1589, 0.7053, 3e-06],
        [34, 0.1451, 0.9498, 0.1587, 0.7075, 3e-06],
        [35, 0.1450, 0.9523, 0.1591, 0.7082, 3e-06],
        [36, 0.1449, 0.9554, 0.1593, 0.7074, 3e-06],
        [37, 0.1448, 0.9576, 0.1592, 0.7050, 3e-06],
        [38, 0.1447, 0.9600, 0.1594, 0.7070, 3e-06],
        [39, 0.1447, 0.9616, 0.1594, 0.7067, 3e-06],
        [40, 0.1446, 0.9631, 0.1601, 0.7024, 3e-06],
        [41, 0.1445, 0.9647, 0.1598, 0.7053, 3e-06],
        [42, 0.1444, 0.9665, 0.1599, 0.7034, 3e-06],
        [43, 0.1444, 0.9682, 0.1603, 0.7029, 3e-06],
        [44, 0.1443, 0.9694, 0.1604, 0.7063, 3e-06],
        [45, 0.1442, 0.9710, 0.1604, 0.7074, 3e-06],
        [46, 0.1442, 0.9719, 0.1606, 0.7064, 3e-06],
        [47, 0.1441, 0.9732, 0.1606, 0.7018, 3e-06],
        [48, 0.1440, 0.9744, 0.1605, 0.7043, 3e-06],
        [49, 0.1440, 0.9751, 0.1609, 0.7048, 3e-06],
    ])
}


def main(results, label):
    best_train_iou = []
    best_val_iou = []

    best_training_loss = []
    best_val_loss = []

    rf_size_arr = []

    for key, value in sorted(results.items()):
        validation_iou_arr = value[:, 4]
        train_iou_arr = value[:, 2]

        validation_loss_arr = value[:, 3]
        train_loss_arr = value[:, 1]

        rf_size_arr.append(key)
        best_train_iou.append(np.max(train_iou_arr))
        best_val_iou.append(np.max(validation_iou_arr))
        best_training_loss.append(np.min(train_loss_arr))
        best_val_loss.append(np.min(validation_loss_arr))

    # Plot best Iou vs Loss Weight
    plt.figure("IoU")
    plt.plot(rf_size_arr, best_train_iou, label='train_' + label, marker='x', markersize=10)
    plt.plot(rf_size_arr, best_val_iou, label='val_' + label, marker='x', markersize=10)
    plt.xlabel("Regularization Loss Weight ")
    plt.xscale('log')
    plt.ylabel("IoU")
    plt.title("Iou vs Regularization Weight")
    plt.legend()
    plt.grid(True)

    # Plot lowest loss vs Tau
    plt.figure("Loss")
    plt.plot(rf_size_arr, best_training_loss, label='train_' + label, marker='x', markersize=10)
    plt.plot(rf_size_arr, best_val_loss, label='val_' + label, marker='x', markersize=10)
    plt.xlabel("Regularization Loss Weight ")
    plt.xscale('log')
    plt.ylabel("Loss")
    plt.title("Loss vs Regularization Weight")
    plt.legend()
    plt.grid(True)

    # Plot Individual Loss/Iou Curves
    num_keys = len(results.keys())
    single_dim = np.ceil(np.sqrt(num_keys))

    fig1 = plt.figure()
    fig2 = plt.figure()

    x_label = 'weight'

    for k_idx, key in enumerate(sorted(results.keys())):
        ax1 = fig1.add_subplot(single_dim, single_dim, k_idx + 1)
        ax2 = fig2.add_subplot(single_dim, single_dim, k_idx + 1)

        ax1.plot(
            results[key][:, 0],
            results[key][:, 2],
            label='train_iou')

        ax1.plot(
            results[key][:, 0],
            results[key][:, 4],
            label='val_iou')

        ax1.set_title(x_label + " {}".format(key))

        ax2.plot(
            results[key][:, 0],
            results[key][:, 1],
            label='train_loss')

        ax2.plot(
            results[key][:, 0],
            results[key][:, 3],
            label='val_iou_loss')

        # ax2.set_yscale('log')

        ax1.set_title(x_label + " = {}".format(key))
        ax1.grid()
        ax2.set_title(x_label + " ={}".format(key))
        ax2.grid()
        ax1.legend()
        ax2.legend()

    fig1.suptitle("{} - Iou Vs Regularization Weight - Individual Curves".format(label))
    fig2.suptitle("{} - Loss Vs Regularization Weight - Individual Curves".format(label))


if __name__ == "__main__":
    plt.ion()

    main(gaussian_reg_weight_results, 'gaussian_reg_sigma_6')
    main(l1_reg_weight_results, 'l1_reg')
    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()

