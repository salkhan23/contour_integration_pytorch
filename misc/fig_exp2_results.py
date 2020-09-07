import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 10,
})

# ---------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------
# results/biped/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base
model_base_vs_time = [
    [1, 0.3648, ['0.18', '0.25', '0.28', '0.26', '0.19', '0.07', '0.01', '0.00'], 0.2991, ['0.31', '0.37', '0.38', '0.37', '0.29', '0.10', '0.01', '0.00'], 0.001],
    [2, 0.2380, ['0.32', '0.37', '0.38', '0.36', '0.30', '0.16', '0.04', '0.01'], 0.2759, ['0.36', '0.40', '0.40', '0.38', '0.33', '0.23', '0.11', '0.02'], 0.001],
    [3, 0.2278, ['0.33', '0.39', '0.40', '0.37', '0.32', '0.23', '0.10', '0.02'], 0.2699, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.25', '0.12', '0.03'], 0.001],
    [4, 0.2236, ['0.34', '0.39', '0.40', '0.38', '0.33', '0.25', '0.13', '0.03'], 0.2691, ['0.34', '0.40', '0.42', '0.41', '0.37', '0.30', '0.17', '0.05'], 0.001],
    [5, 0.2215, ['0.34', '0.40', '0.41', '0.39', '0.34', '0.26', '0.14', '0.04'], 0.2591, ['0.37', '0.42', '0.43', '0.42', '0.38', '0.31', '0.19', '0.07'], 0.001],
    [6, 0.2184, ['0.34', '0.40', '0.41', '0.39', '0.34', '0.27', '0.15', '0.05'], 0.2606, ['0.37', '0.42', '0.43', '0.41', '0.36', '0.27', '0.15', '0.04'], 0.001],
    [7, 0.2183, ['0.35', '0.40', '0.41', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2602, ['0.37', '0.42', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.001],
    [8, 0.2203, ['0.35', '0.40', '0.41', '0.40', '0.35', '0.27', '0.16', '0.06'], 0.2645, ['0.38', '0.42', '0.43', '0.41', '0.36', '0.28', '0.17', '0.06'], 0.001],
    [9, 0.2177, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2598, ['0.38', '0.43', '0.43', '0.41', '0.36', '0.27', '0.16', '0.05'], 0.001],
    [10, 0.2157, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.06'], 0.2547, ['0.36', '0.42', '0.44', '0.44', '0.40', '0.33', '0.22', '0.10'], 0.001],
    [11, 0.2157, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.18', '0.07'], 0.2627, ['0.38', '0.42', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.001],
    [12, 0.2145, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.18', '0.07'], 0.2562, ['0.38', '0.43', '0.44', '0.43', '0.39', '0.31', '0.21', '0.09'], 0.001],
    [13, 0.2138, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2564, ['0.36', '0.42', '0.44', '0.43', '0.40', '0.33', '0.22', '0.11'], 0.001],
    [14, 0.2170, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.18', '0.07'], 0.2590, ['0.35', '0.41', '0.44', '0.43', '0.40', '0.33', '0.22', '0.10'], 0.001],
    [15, 0.2135, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.29', '0.18', '0.07'], 0.2592, ['0.39', '0.43', '0.44', '0.42', '0.37', '0.29', '0.18', '0.07'], 0.001],
    [16, 0.2128, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.29', '0.18', '0.07'], 0.2534, ['0.37', '0.43', '0.44', '0.43', '0.39', '0.31', '0.21', '0.09'], 0.001],
    [17, 0.2124, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.29', '0.19', '0.08'], 0.2621, ['0.38', '0.43', '0.43', '0.41', '0.36', '0.29', '0.19', '0.09'], 0.001],
    [18, 0.2123, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.29', '0.19', '0.08'], 0.2556, ['0.37', '0.43', '0.44', '0.43', '0.39', '0.32', '0.21', '0.10'], 0.001],
    [19, 0.2127, ['0.35', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2617, ['0.38', '0.42', '0.43', '0.41', '0.37', '0.30', '0.20', '0.09'], 0.001],
    [20, 0.2131, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2642, ['0.39', '0.43', '0.43', '0.41', '0.36', '0.29', '0.19', '0.09'], 0.001],
    [21, 0.2118, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2522, ['0.36', '0.42', '0.45', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.001],
    [22, 0.2116, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2565, ['0.37', '0.43', '0.44', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.001],
    [23, 0.2115, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2553, ['0.35', '0.42', '0.44', '0.44', '0.41', '0.34', '0.23', '0.11'], 0.001],
    [24, 0.2129, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2605, ['0.39', '0.43', '0.44', '0.41', '0.36', '0.28', '0.18', '0.08'], 0.001],
    [25, 0.2112, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2529, ['0.35', '0.41', '0.44', '0.44', '0.41', '0.34', '0.23', '0.11'], 0.001],
    [26, 0.2116, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2575, ['0.39', '0.44', '0.44', '0.41', '0.37', '0.29', '0.19', '0.09'], 0.001],
    [27, 0.2115, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.19', '0.09'], 0.2492, ['0.37', '0.43', '0.45', '0.44', '0.39', '0.32', '0.21', '0.10'], 0.001],
    [28, 0.2110, ['0.36', '0.41', '0.43', '0.42', '0.37', '0.30', '0.20', '0.09'], 0.2533, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.001],
    [29, 0.2106, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.20', '0.09'], 0.2499, ['0.37', '0.43', '0.45', '0.44', '0.40', '0.32', '0.22', '0.10'], 0.001],
    [30, 0.2105, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.20', '0.09'], 0.2526, ['0.38', '0.43', '0.45', '0.43', '0.39', '0.31', '0.21', '0.09'], 0.001],
    [31, 0.2102, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.20', '0.09'], 0.2524, ['0.38', '0.43', '0.45', '0.43', '0.39', '0.31', '0.20', '0.09'], 0.001],
    [32, 0.1984, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.30', '0.20', '0.09'], 0.2399, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.33', '0.23', '0.11'], 0.0001],
    [33, 0.1973, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2411, ['0.39', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.11'], 0.0001],
    [34, 0.1972, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2386, ['0.37', '0.43', '0.45', '0.45', '0.42', '0.35', '0.25', '0.13'], 0.0001],
    [35, 0.1971, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2411, ['0.39', '0.44', '0.45', '0.44', '0.40', '0.32', '0.22', '0.10'], 0.0001],
    [36, 0.1970, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2402, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.12'], 0.0001],
    [37, 0.1969, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2393, ['0.38', '0.43', '0.45', '0.45', '0.41', '0.34', '0.24', '0.12'], 0.0001],
    [38, 0.1969, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2399, ['0.38', '0.43', '0.45', '0.44', '0.41', '0.34', '0.24', '0.12'], 0.0001],
    [39, 0.1967, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2409, ['0.39', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.11'], 0.0001],
    [40, 0.1967, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2381, ['0.38', '0.43', '0.46', '0.45', '0.42', '0.35', '0.25', '0.13'], 0.0001],
    [41, 0.1966, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2400, ['0.38', '0.43', '0.45', '0.45', '0.41', '0.35', '0.25', '0.13'], 0.0001],
    [42, 0.1966, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2426, ['0.39', '0.44', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.0001],
    [43, 0.1965, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2394, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.34', '0.24', '0.12'], 0.0001],
    [44, 0.1964, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2400, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.12'], 0.0001],
    [45, 0.1964, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2389, ['0.38', '0.43', '0.45', '0.45', '0.41', '0.35', '0.25', '0.13'], 0.0001],
    [46, 0.1963, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2400, ['0.39', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.12'], 0.0001],
    [47, 0.1962, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2396, ['0.39', '0.44', '0.45', '0.44', '0.41', '0.34', '0.24', '0.13'], 0.0001],
    [48, 0.1962, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2398, ['0.38', '0.44', '0.45', '0.44', '0.41', '0.34', '0.25', '0.13'], 0.0001],
    [49, 0.1961, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2390, ['0.38', '0.43', '0.45', '0.45', '0.41', '0.34', '0.24', '0.12'], 0.0001],
    [50, 0.1961, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.10'], 0.2411, ['0.39', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.12'], 0.0001]
]
control_base_vs_time = [
    [1, 0.3943, ['0.18', '0.24', '0.28', '0.30', '0.27', '0.03', '0.00', '0.00'], 0.2913, ['0.32', '0.37', '0.40', '0.40', '0.36', '0.12', '0.01', '0.00'], 0.001],
    [2, 0.2368, ['0.32', '0.38', '0.39', '0.37', '0.32', '0.18', '0.06', '0.01'], 0.2743, ['0.37', '0.41', '0.42', '0.41', '0.38', '0.30', '0.17', '0.05'], 0.001],
    [3, 0.2296, ['0.34', '0.39', '0.40', '0.38', '0.33', '0.24', '0.11', '0.03'], 0.2774, ['0.38', '0.42', '0.42', '0.39', '0.33', '0.23', '0.12', '0.03'], 0.001],
    [4, 0.2276, ['0.34', '0.40', '0.41', '0.39', '0.34', '0.25', '0.13', '0.04'], 0.2645, ['0.37', '0.42', '0.43', '0.42', '0.37', '0.28', '0.16', '0.05'], 0.001],
    [5, 0.2263, ['0.35', '0.40', '0.41', '0.39', '0.34', '0.26', '0.14', '0.04'], 0.2624, ['0.38', '0.42', '0.44', '0.43', '0.39', '0.32', '0.20', '0.07'], 0.001],
    [6, 0.2247, ['0.35', '0.40', '0.41', '0.40', '0.35', '0.26', '0.15', '0.04'], 0.2663, ['0.39', '0.43', '0.44', '0.41', '0.37', '0.28', '0.17', '0.05'], 0.001],
    [7, 0.2239, ['0.35', '0.40', '0.41', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2749, ['0.38', '0.42', '0.42', '0.40', '0.34', '0.26', '0.14', '0.04'], 0.001],
    [8, 0.2240, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2676, ['0.41', '0.44', '0.43', '0.40', '0.36', '0.28', '0.17', '0.07'], 0.001],
    [9, 0.2236, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2600, ['0.39', '0.43', '0.44', '0.42', '0.37', '0.29', '0.17', '0.05'], 0.001],
    [10, 0.2228, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.05'], 0.2551, ['0.36', '0.42', '0.45', '0.44', '0.41', '0.33', '0.21', '0.07'], 0.001],
    [11, 0.2220, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2575, ['0.38', '0.43', '0.45', '0.43', '0.39', '0.31', '0.20', '0.07'], 0.001],
    [12, 0.2218, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2610, ['0.38', '0.43', '0.44', '0.43', '0.39', '0.31', '0.19', '0.07'], 0.001],
    [13, 0.2215, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2646, ['0.39', '0.43', '0.44', '0.41', '0.36', '0.27', '0.16', '0.05'], 0.001],
    [14, 0.2220, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2544, ['0.37', '0.43', '0.45', '0.45', '0.42', '0.36', '0.25', '0.10'], 0.001],
    [15, 0.2216, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2571, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.33', '0.21', '0.08'], 0.001],
    [16, 0.2212, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.06'], 0.2609, ['0.38', '0.43', '0.44', '0.43', '0.39', '0.32', '0.21', '0.08'], 0.001],
    [17, 0.2210, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.06'], 0.2585, ['0.38', '0.43', '0.44', '0.43', '0.38', '0.30', '0.19', '0.07'], 0.001],
    [18, 0.2204, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2600, ['0.36', '0.41', '0.44', '0.45', '0.43', '0.37', '0.27', '0.13'], 0.001],
    [19, 0.2208, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2636, ['0.39', '0.43', '0.44', '0.42', '0.37', '0.29', '0.18', '0.07'], 0.001],
    [20, 0.2206, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2554, ['0.37', '0.43', '0.45', '0.44', '0.40', '0.33', '0.22', '0.09'], 0.001],
    [21, 0.2207, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2583, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.32', '0.21', '0.08'], 0.001],
    [22, 0.2202, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2556, ['0.36', '0.42', '0.45', '0.44', '0.40', '0.32', '0.20', '0.08'], 0.001],
    [23, 0.2201, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2592, ['0.35', '0.42', '0.44', '0.44', '0.41', '0.33', '0.22', '0.09'], 0.001],
    [24, 0.2204, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2564, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.07'], 0.001],
    [25, 0.2199, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.17', '0.06'], 0.2543, ['0.37', '0.42', '0.45', '0.45', '0.42', '0.35', '0.23', '0.09'], 0.001],
    [26, 0.2199, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.17', '0.06'], 0.2561, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.33', '0.22', '0.09'], 0.001],
    [27, 0.2196, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.06'], 0.2547, ['0.37', '0.43', '0.45', '0.43', '0.38', '0.29', '0.18', '0.07'], 0.001],
    [28, 0.2198, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2554, ['0.36', '0.42', '0.45', '0.44', '0.41', '0.33', '0.22', '0.08'], 0.001],
    [29, 0.2194, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2549, ['0.38', '0.44', '0.45', '0.44', '0.39', '0.30', '0.19', '0.07'], 0.001],
    [30, 0.2197, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2646, ['0.33', '0.40', '0.44', '0.44', '0.41', '0.34', '0.23', '0.09'], 0.001],
    [31, 0.2202, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2617, ['0.39', '0.43', '0.44', '0.42', '0.37', '0.30', '0.20', '0.08'], 0.001],
    [32, 0.2041, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.18', '0.07'], 0.2398, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.22', '0.09'], 0.0001],
    [33, 0.2033, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.18', '0.07'], 0.2392, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.0001],
    [34, 0.2031, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.2386, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.33', '0.22', '0.09'], 0.0001],
    [35, 0.2030, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.2381, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.33', '0.22', '0.10'], 0.0001],
    [36, 0.2028, ['0.36', '0.42', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.2374, ['0.38', '0.44', '0.46', '0.45', '0.41', '0.33', '0.22', '0.09'], 0.0001],
    [37, 0.2028, ['0.36', '0.42', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2384, ['0.39', '0.44', '0.46', '0.45', '0.40', '0.33', '0.22', '0.09'], 0.0001],
    [38, 0.2027, ['0.36', '0.42', '0.43', '0.41', '0.37', '0.29', '0.19', '0.08'], 0.2383, ['0.39', '0.44', '0.46', '0.45', '0.40', '0.33', '0.22', '0.09'], 0.0001],
    [39, 0.2026, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2390, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.0001],
    [40, 0.2025, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2376, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.34', '0.24', '0.11'], 0.0001],
    [41, 0.2024, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2386, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.0001],
    [42, 0.2023, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2396, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.22', '0.09'], 0.0001],
    [43, 0.2023, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2383, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.31', '0.20', '0.08'], 0.0001],
    [44, 0.2023, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2395, ['0.39', '0.44', '0.46', '0.45', '0.40', '0.33', '0.22', '0.10'], 0.0001],
    [45, 0.2022, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2390, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.0001],
    [46, 0.2021, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2383, ['0.38', '0.43', '0.46', '0.45', '0.41', '0.33', '0.22', '0.09'], 0.0001],
    [47, 0.2021, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2380, ['0.38', '0.44', '0.46', '0.45', '0.41', '0.34', '0.23', '0.10'], 0.0001],
    [48, 0.2020, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2374, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.33', '0.23', '0.10'], 0.0001],
    [49, 0.2019, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2368, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.34', '0.23', '0.10'], 0.0001],
    [50, 0.2019, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.29', '0.19', '0.08'], 0.2393, ['0.39', '0.44', '0.46', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
]
# Edges from entire validation set (50 images)
edge_counts_base = np.array([  # above, below, on diagonal
    [20317., 15157., 3291.],
    [19893., 17066., 885.],
    [22202., 20237., 771.],
    [23287., 21780., 597.],
    [26577., 24711., 680.],
    [28446., 26884., 679.],
    [28955., 27756., 787.],
    [28067., 29152., 995.],
    [20532., 24262., 1169.],
    [4937., 11341., 615.]])
non_edge_counts_base = np.array([
    [5.21102e+05, 4.70762e+05, 1.20536e+06],
    [8.98890e+04, 1.17857e+05, 5.32500e+03],
    [5.56600e+04, 7.63370e+04, 2.53100e+03],
    [3.75380e+04, 5.20730e+04, 1.22800e+03],
    [2.83140e+04, 4.02810e+04, 9.14000e+02],
    [2.05740e+04, 2.94450e+04, 6.86000e+02],
    [1.29310e+04, 2.09420e+04, 4.57000e+02],
    [7.96100e+03, 1.33390e+04, 3.74000e+02],
    [3.51400e+03, 6.73000e+03, 2.11000e+02],
    [4.51000e+02, 1.73500e+03, 8.10000e+01]])

model_3x3_vs_time = [
    [1, 0.3417, ['0.19', '0.26', '0.31', '0.33', '0.30', '0.13', '0.04', '0.01'], 0.2647, ['0.33', '0.40', '0.41', '0.40', '0.35', '0.25', '0.10', '0.02'], 0.001],
    [2, 0.2156, ['0.33', '0.39', '0.40', '0.38', '0.33', '0.24', '0.11', '0.03'], 0.2503, ['0.36', '0.41', '0.43', '0.41', '0.36', '0.27', '0.15', '0.05'], 0.001],
    [3, 0.2072, ['0.34', '0.40', '0.41', '0.39', '0.35', '0.27', '0.15', '0.04'], 0.2430, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.29', '0.17', '0.06'], 0.001],
    [4, 0.2040, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2405, ['0.36', '0.42', '0.44', '0.44', '0.41', '0.34', '0.23', '0.10'], 0.001],
    [5, 0.2019, ['0.35', '0.41', '0.42', '0.40', '0.36', '0.28', '0.17', '0.06'], 0.2405, ['0.38', '0.43', '0.44', '0.43', '0.39', '0.31', '0.21', '0.09'], 0.001],
    [6, 0.2005, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.29', '0.18', '0.07'], 0.2407, ['0.38', '0.43', '0.44', '0.43', '0.38', '0.30', '0.19', '0.07'], 0.001],
    [7, 0.1994, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.18', '0.07'], 0.2393, ['0.38', '0.43', '0.45', '0.43', '0.39', '0.31', '0.20', '0.09'], 0.001],
    [8, 0.1986, ['0.36', '0.41', '0.43', '0.41', '0.37', '0.29', '0.19', '0.07'], 0.2373, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.33', '0.23', '0.10'], 0.001],
    [9, 0.1978, ['0.36', '0.42', '0.43', '0.41', '0.37', '0.30', '0.19', '0.07'], 0.2362, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.33', '0.22', '0.10'], 0.001],
    [10, 0.1973, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.19', '0.08'], 0.2375, ['0.38', '0.43', '0.45', '0.44', '0.41', '0.35', '0.25', '0.12'], 0.001],
    [11, 0.1969, ['0.36', '0.42', '0.43', '0.42', '0.37', '0.30', '0.19', '0.08'], 0.2416, ['0.40', '0.44', '0.45', '0.42', '0.37', '0.29', '0.18', '0.07'], 0.001],
    [12, 0.1963, ['0.36', '0.42', '0.43', '0.42', '0.38', '0.30', '0.20', '0.08'], 0.2417, ['0.39', '0.44', '0.45', '0.43', '0.40', '0.33', '0.24', '0.12'], 0.001],
    [13, 0.1961, ['0.36', '0.42', '0.43', '0.42', '0.38', '0.30', '0.20', '0.08'], 0.2378, ['0.40', '0.44', '0.45', '0.44', '0.39', '0.32', '0.21', '0.09'], 0.001],
    [14, 0.1956, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.30', '0.20', '0.08'], 0.2355, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.11'], 0.001],
    [15, 0.1954, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.20', '0.08'], 0.2322, ['0.38', '0.44', '0.46', '0.45', '0.42', '0.35', '0.24', '0.10'], 0.001],
    [16, 0.1951, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.20', '0.09'], 0.2346, ['0.39', '0.44', '0.46', '0.44', '0.40', '0.34', '0.23', '0.11'], 0.001],
    [17, 0.1946, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.21', '0.09'], 0.2380, ['0.38', '0.43', '0.45', '0.44', '0.41', '0.35', '0.25', '0.13'], 0.001],
    [18, 0.1945, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.21', '0.09'], 0.2349, ['0.38', '0.44', '0.45', '0.44', '0.40', '0.33', '0.23', '0.11'], 0.001],
    [19, 0.1943, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.21', '0.09'], 0.2325, ['0.37', '0.43', '0.46', '0.46', '0.42', '0.36', '0.25', '0.10'], 0.001],
    [20, 0.1942, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.21', '0.09'], 0.2374, ['0.40', '0.44', '0.45', '0.44', '0.40', '0.34', '0.24', '0.11'], 0.001],
    [21, 0.1941, ['0.36', '0.42', '0.44', '0.42', '0.38', '0.31', '0.21', '0.09'], 0.2384, ['0.39', '0.44', '0.45', '0.44', '0.39', '0.32', '0.22', '0.10'], 0.001],
    [22, 0.1939, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2334, ['0.39', '0.44', '0.46', '0.45', '0.41', '0.34', '0.23', '0.11'], 0.001],
    [23, 0.1938, ['0.36', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2363, ['0.39', '0.44', '0.45', '0.45', '0.41', '0.35', '0.26', '0.13'], 0.001],
    [24, 0.1936, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2338, ['0.37', '0.43', '0.45', '0.45', '0.42', '0.35', '0.25', '0.12'], 0.001],
    [25, 0.1935, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2338, ['0.38', '0.44', '0.46', '0.45', '0.42', '0.36', '0.27', '0.14'], 0.001],
    [26, 0.1933, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2353, ['0.35', '0.42', '0.45', '0.46', '0.44', '0.39', '0.30', '0.17'], 0.001],
    [27, 0.1932, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2350, ['0.39', '0.44', '0.46', '0.44', '0.41', '0.34', '0.24', '0.11'], 0.001],
    [28, 0.1932, ['0.37', '0.42', '0.44', '0.43', '0.38', '0.31', '0.21', '0.09'], 0.2350, ['0.38', '0.44', '0.45', '0.45', '0.42', '0.36', '0.26', '0.14'], 0.001],
    [29, 0.1930, ['0.37', '0.42', '0.44', '0.43', '0.39', '0.31', '0.21', '0.10'], 0.2342, ['0.38', '0.44', '0.46', '0.45', '0.41', '0.34', '0.24', '0.11'], 0.001],
    [30, 0.1930, ['0.37', '0.42', '0.44', '0.43', '0.39', '0.31', '0.21', '0.10'], 0.2367, ['0.40', '0.45', '0.45', '0.44', '0.39', '0.33', '0.23', '0.12'], 0.001],
    [31, 0.1929, ['0.37', '0.42', '0.44', '0.43', '0.39', '0.31', '0.21', '0.10'], 0.2296, ['0.38', '0.44', '0.46', '0.46', '0.43', '0.36', '0.26', '0.13'], 0.001],
    [32, 0.1910, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2317, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.35', '0.26', '0.13'], 0.0001],
    [33, 0.1907, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2310, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.13'], 0.0001],
    [34, 0.1907, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2312, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.14'], 0.0001],
    [35, 0.1907, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2315, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.35', '0.26', '0.13'], 0.0001],
    [36, 0.1906, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2315, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.35', '0.26', '0.14'], 0.0001],
    [37, 0.1906, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2314, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.35', '0.25', '0.13'], 0.0001],
    [38, 0.1905, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2326, ['0.39', '0.45', '0.46', '0.45', '0.41', '0.35', '0.25', '0.13'], 0.0001],
    [39, 0.1905, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2312, ['0.39', '0.45', '0.46', '0.45', '0.42', '0.35', '0.25', '0.13'], 0.0001],
    [40, 0.1904, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2308, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.14'], 0.0001],
    [41, 0.1904, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2306, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.13'], 0.0001],
    [42, 0.1904, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2327, ['0.39', '0.45', '0.46', '0.45', '0.41', '0.35', '0.25', '0.13'], 0.0001],
    [43, 0.1904, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2303, ['0.39', '0.44', '0.46', '0.46', '0.42', '0.36', '0.27', '0.14'], 0.0001],
    [44, 0.1904, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.10'], 0.2309, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.14'], 0.0001],
    [45, 0.1903, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2308, ['0.39', '0.44', '0.46', '0.46', '0.43', '0.36', '0.27', '0.14'], 0.0001],
    [46, 0.1903, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2318, ['0.39', '0.45', '0.46', '0.45', '0.41', '0.35', '0.25', '0.13'], 0.0001],
    [47, 0.1902, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2306, ['0.39', '0.44', '0.46', '0.45', '0.42', '0.36', '0.26', '0.13'], 0.0001],
    [48, 0.1903, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2305, ['0.39', '0.44', '0.46', '0.46', '0.42', '0.36', '0.27', '0.14'], 0.0001],
    [49, 0.1902, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2305, ['0.39', '0.45', '0.46', '0.45', '0.42', '0.35', '0.26', '0.13'], 0.0001],
    [50, 0.1902, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.32', '0.22', '0.11'], 0.2303, ['0.39', '0.44', '0.46', '0.46', '0.42', '0.36', '0.27', '0.14'], 0.0001],
]

control_3x3_vs_time = [
    [1, 0.3543, ['0.18', '0.24', '0.29', '0.32', '0.27', '0.11', '0.04', '0.01'], 0.2594, ['0.34', '0.40', '0.42', '0.42', '0.37', '0.24', '0.11', '0.03'], 0.001],
    [2, 0.2157, ['0.33', '0.39', '0.40', '0.38', '0.33', '0.22', '0.10', '0.02'], 0.2468, ['0.36', '0.41', '0.43', '0.42', '0.39', '0.30', '0.16', '0.05'], 0.001],
    [3, 0.2098, ['0.34', '0.39', '0.41', '0.39', '0.34', '0.25', '0.13', '0.03'], 0.2431, ['0.37', '0.42', '0.44', '0.42', '0.38', '0.29', '0.16', '0.05'], 0.001],
    [4, 0.2077, ['0.34', '0.40', '0.41', '0.39', '0.34', '0.26', '0.14', '0.04'], 0.2457, ['0.38', '0.42', '0.43', '0.41', '0.35', '0.25', '0.13', '0.03'], 0.001],
    [5, 0.2066, ['0.34', '0.40', '0.41', '0.39', '0.34', '0.26', '0.15', '0.05'], 0.2404, ['0.38', '0.43', '0.44', '0.43', '0.38', '0.29', '0.18', '0.07'], 0.001],
    [6, 0.2058, ['0.35', '0.40', '0.41', '0.39', '0.34', '0.26', '0.15', '0.05'], 0.2428, ['0.38', '0.43', '0.44', '0.42', '0.37', '0.28', '0.17', '0.06'], 0.001],
    [7, 0.2054, ['0.35', '0.40', '0.41', '0.40', '0.35', '0.26', '0.16', '0.05'], 0.2404, ['0.36', '0.42', '0.44', '0.43', '0.39', '0.31', '0.20', '0.07'], 0.001],
    [8, 0.2050, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.05'], 0.2397, ['0.39', '0.43', '0.44', '0.43', '0.38', '0.30', '0.18', '0.06'], 0.001],
    [9, 0.2045, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.06'], 0.2389, ['0.39', '0.43', '0.45', '0.43', '0.38', '0.29', '0.17', '0.06'], 0.001],
    [10, 0.2042, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.06'], 0.2392, ['0.39', '0.43', '0.45', '0.43', '0.38', '0.30', '0.18', '0.07'], 0.001],
    [11, 0.2040, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.16', '0.06'], 0.2397, ['0.38', '0.43', '0.44', '0.43', '0.38', '0.30', '0.20', '0.08'], 0.001],
    [12, 0.2038, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2393, ['0.38', '0.43', '0.44', '0.43', '0.39', '0.31', '0.20', '0.08'], 0.001],
    [13, 0.2036, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2405, ['0.38', '0.43', '0.44', '0.42', '0.37', '0.29', '0.18', '0.07'], 0.001],
    [14, 0.2035, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2463, ['0.39', '0.43', '0.43', '0.40', '0.34', '0.25', '0.14', '0.05'], 0.001],
    [15, 0.2034, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2415, ['0.39', '0.43', '0.44', '0.42', '0.36', '0.28', '0.17', '0.07'], 0.001],
    [16, 0.2032, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2382, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.29', '0.18', '0.06'], 0.001],
    [17, 0.2032, ['0.35', '0.40', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2404, ['0.38', '0.43', '0.44', '0.42', '0.37', '0.29', '0.18', '0.07'], 0.001],
    [18, 0.2030, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2408, ['0.38', '0.43', '0.44', '0.43', '0.38', '0.30', '0.19', '0.08'], 0.001],
    [19, 0.2030, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2374, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.30', '0.19', '0.07'], 0.001],
    [20, 0.2029, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2383, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.29', '0.18', '0.07'], 0.001],
    [21, 0.2029, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2387, ['0.37', '0.43', '0.45', '0.43', '0.39', '0.31', '0.20', '0.08'], 0.001],
    [22, 0.2027, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2383, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.31', '0.20', '0.08'], 0.001],
    [23, 0.2027, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.06'], 0.2370, ['0.37', '0.43', '0.45', '0.44', '0.40', '0.32', '0.21', '0.08'], 0.001],
    [24, 0.2027, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.07'], 0.2399, ['0.39', '0.43', '0.45', '0.43', '0.38', '0.30', '0.20', '0.09'], 0.001],
    [25, 0.2026, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.27', '0.17', '0.07'], 0.2371, ['0.37', '0.43', '0.45', '0.43', '0.38', '0.29', '0.17', '0.06'], 0.001],
    [26, 0.2025, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2371, ['0.37', '0.43', '0.45', '0.44', '0.40', '0.33', '0.21', '0.09'], 0.001],
    [27, 0.2025, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2363, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.30', '0.18', '0.06'], 0.001],
    [28, 0.2025, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2393, ['0.39', '0.43', '0.45', '0.43', '0.38', '0.30', '0.19', '0.07'], 0.001],
    [29, 0.2023, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2355, ['0.37', '0.43', '0.45', '0.44', '0.40', '0.31', '0.20', '0.08'], 0.001],
    [30, 0.2024, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2380, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.30', '0.18', '0.07'], 0.001],
    [31, 0.2023, ['0.35', '0.41', '0.42', '0.40', '0.35', '0.28', '0.17', '0.07'], 0.2365, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.32', '0.21', '0.09'], 0.001],
    [32, 0.2010, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2363, ['0.38', '0.44', '0.45', '0.43', '0.39', '0.30', '0.19', '0.07'], 0.0001],
    [33, 0.2007, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2356, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [34, 0.2007, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2356, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [35, 0.2007, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2358, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [36, 0.2006, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2359, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [37, 0.2006, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2358, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [38, 0.2005, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2355, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [39, 0.2005, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2362, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.19', '0.08'], 0.0001],
    [40, 0.2005, ['0.35', '0.41', '0.42', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2362, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [41, 0.2004, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2370, ['0.38', '0.43', '0.45', '0.43', '0.38', '0.30', '0.19', '0.08'], 0.0001],
    [42, 0.2005, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2359, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [43, 0.2004, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2355, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.32', '0.21', '0.09'], 0.0001],
    [44, 0.2004, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2347, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.32', '0.21', '0.08'], 0.0001],
    [45, 0.2004, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2345, ['0.38', '0.43', '0.45', '0.44', '0.40', '0.32', '0.21', '0.09'], 0.0001],
    [46, 0.2003, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2353, ['0.38', '0.44', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [47, 0.2003, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2356, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.19', '0.07'], 0.0001],
    [48, 0.2003, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2352, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
    [49, 0.2003, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2356, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.32', '0.20', '0.08'], 0.0001],
    [50, 0.2003, ['0.35', '0.41', '0.43', '0.41', '0.36', '0.28', '0.18', '0.07'], 0.2359, ['0.38', '0.43', '0.45', '0.44', '0.39', '0.31', '0.20', '0.08'], 0.0001],
]


# ---------------------------------------------------------------------------------------
def get_epoch_vs_iou_results(results, iou_type='train', iou_th_idx=2):
    """
     From the collected results retrieve the epoch vs iou results

    :param results:
    :param iou_type:
    :param iou_th_idx:
    :return:
    """
    epoch_iou_array = np.zeros((len(results), 3))  # epoch, loss, iou

    if iou_type.lower == 'train':
        iou_type_idx = 2
        loss_idx = 1
    elif iou_type == 'val':
        iou_type_idx = 4
        loss_idx = 3
    else:
        raise Exception("Invalid iou Type !")

    for e_idx, epoch_results in enumerate(results):
        epoch_iou_array[e_idx, 0] = epoch_results[0]
        epoch_iou_array[e_idx, 1] = epoch_results[loss_idx]
        epoch_iou_array[e_idx, 2] = epoch_results[iou_type_idx][iou_th_idx]

    return epoch_iou_array


if __name__ == '__main__':
    plt.ion()

    model_base_time_results = get_epoch_vs_iou_results(model_base_vs_time, iou_type='val')
    control_base_time_results = get_epoch_vs_iou_results(control_base_vs_time, iou_type='val')
    model_3x3_time_results = get_epoch_vs_iou_results(model_3x3_vs_time, iou_type='val')
    control_3x3_time_results = get_epoch_vs_iou_results(control_3x3_vs_time, iou_type='val')

    # -----------------------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------------------
    f = plt.figure(constrained_layout=True, figsize=(12, 9))
    gs = f.add_gridspec(3, 2)

    # Loss vs Epoch
    # --------------
    ax1 = f.add_subplot(gs[0, 0])

    ax1.plot(model_base_time_results[:, 0], model_base_time_results[:, 1], label='model')
    ax1.plot(control_base_time_results[:, 0], control_base_time_results[:, 1], label='control')
    ax1.plot(model_3x3_time_results[:, 0], model_3x3_time_results[:, 1], label='model_3x3')
    ax1.plot(control_3x3_time_results[:, 0], control_3x3_time_results[:, 1], label='control_3x3')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    # ax1.legend()

    # IoU vs Epoch
    # --------------
    ax2 = f.add_subplot(gs[0, 1])

    ax2.plot(model_base_time_results[:, 0], model_base_time_results[:, 2], label='model')
    ax2.plot(control_base_time_results[:, 0], control_base_time_results[:, 2], label='control')
    ax2.plot(model_3x3_time_results[:, 0], model_3x3_time_results[:, 2], label='model_3x3')
    ax2.plot(control_3x3_time_results[:, 0], control_3x3_time_results[:, 2], label='control_3x3')

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()

    # Summary of scatter plots edges model vs control
    # -----------------------------------------------------
    edge_strength_bins = np.arange(0.1, 1.1, 0.1)

    ax3 = f.add_subplot(gs[1, 0])

    ax3.plot(edge_strength_bins, edge_counts_base[:, 0]/1000., label='above')
    ax3.plot(edge_strength_bins, edge_counts_base[:, 1]/1000., label='below')
    ax3.plot(edge_strength_bins, edge_counts_base[:, 2]/1000., label='on')

    ax3.set_title("Edges")
    ax3.set_xlabel('Prediction')
    ax3.set_ylabel('Count (10K)')
    # ax3.legend()

    # Summary of scatter plots non-edges model vs control
    # -----------------------------------------------------
    edge_strength_bins = np.arange(0.1, 1.1, 0.1)

    ax4 = f.add_subplot(gs[1, 1])

    ax4.plot(edge_strength_bins, non_edge_counts_base[:, 0] / 1000., label='above')
    ax4.plot(edge_strength_bins, non_edge_counts_base[:, 1] / 1000., label='below')
    ax4.plot(edge_strength_bins, non_edge_counts_base[:, 2] / 1000., label='on')

    ax4.set_title("Non-Edges")
    ax4.set_xlabel('Prediction')
    ax4.set_ylabel('Count (10K)')
    ax4.legend()
    ax4.set_ylim([0, 500])

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print('End')
    import pdb

    pdb.set_trace()
