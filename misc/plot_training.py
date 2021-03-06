import numpy as np
import matplotlib.pyplot as plt

# Copy this from the summary File
# Current Subtractive
data1 = np.array([
    [0,  0.7080, 0.8316, 0.6850, 0.8977, 0.001],
    [1,  0.6849, 0.9165, 0.6846, 0.9204, 0.001],
    [2,  0.6846, 0.9322, 0.6845, 0.9401, 0.001],
    [3,  0.6845, 0.9410, 0.6845, 0.9371, 0.001],
    [4,  0.6844, 0.9473, 0.6844, 0.9477, 0.001],
    [5,  0.6844, 0.9502, 0.6844, 0.9513, 0.001],
    [6,  0.6843, 0.9543, 0.6845, 0.9377, 0.001],
    [7,  0.6843, 0.9559, 0.6843, 0.9572, 0.001],
    [8,  0.6843, 0.9590, 0.6843, 0.9535, 0.001],
    [9,  0.6843, 0.9600, 0.6843, 0.9559, 0.001],
    [10,  0.6843, 0.9618, 0.6843, 0.9594, 0.001],
    [11,  0.6843, 0.9635, 0.6843, 0.9560, 0.001],
    [12,  0.6842, 0.9640, 0.6843, 0.9579, 0.001],
    [13,  0.6842, 0.9658, 0.6843, 0.9600, 0.001],
    [14,  0.6842, 0.9662, 0.6843, 0.9607, 0.001],
    [15,  0.6842, 0.9671, 0.6843, 0.9557, 0.001],
    [16,  0.6842, 0.9681, 0.6843, 0.9637, 0.001],
    [17,  0.6842, 0.9682, 0.6842, 0.9613, 0.001],
    [18,  0.6842, 0.9699, 0.6843, 0.9594, 0.001],
    [19,  0.6842, 0.9708, 0.6843, 0.9557, 0.001],
    [20,  0.6842, 0.9719, 0.6842, 0.9630, 0.001],
    [21,  0.6842, 0.9719, 0.6843, 0.9606, 0.001],
    [22,  0.6842, 0.9728, 0.6842, 0.9656, 0.001],
    [23,  0.6842, 0.9734, 0.6842, 0.9622, 0.001],
    [24,  0.6842, 0.9742, 0.6842, 0.9631, 0.001],
    [25,  0.6841, 0.9746, 0.6842, 0.9658, 0.001],
    [26,  0.6841, 0.9753, 0.6842, 0.9659, 0.001],
    [27,  0.6841, 0.9752, 0.6842, 0.9672, 0.001],
    [28,  0.6841, 0.9760, 0.6842, 0.9652, 0.001],
    [29,  0.6841, 0.9760, 0.6842, 0.9658, 0.001],
    [30,  0.6841, 0.9763, 0.6842, 0.9631, 0.001],
    [31,  0.6841, 0.9764, 0.6842, 0.9670, 0.001],
    [32,  0.6841, 0.9782, 0.6843, 0.9594, 0.001],
    [33,  0.6841, 0.9780, 0.6842, 0.9673, 0.001],
    [34,  0.6841, 0.9782, 0.6842, 0.9657, 0.001],
    [35,  0.6841, 0.9787, 0.6842, 0.9676, 0.001],
    [36,  0.6841, 0.9788, 0.6842, 0.9688, 0.001],
    [37,  0.6841, 0.9796, 0.6842, 0.9659, 0.001],
    [38,  0.6841, 0.9793, 0.6842, 0.9680, 0.001],
    [39,  0.6841, 0.9796, 0.6842, 0.9658, 0.001],
    [40,  0.6841, 0.9802, 0.6842, 0.9681, 0.001],
    [41,  0.6841, 0.9804, 0.6842, 0.9677, 0.001],
    [42,  0.6841, 0.9804, 0.6842, 0.9701, 0.001],
    [43,  0.6841, 0.9809, 0.6842, 0.9674, 0.001],
    [44,  0.6841, 0.9807, 0.6842, 0.9654, 0.001],
    [45,  0.6841, 0.9812, 0.6842, 0.9669, 0.001],
    [46,  0.6841, 0.9815, 0.6842, 0.9671, 0.001],
    [47,  0.6841, 0.9821, 0.6842, 0.9699, 0.001],
    [48,  0.6841, 0.9818, 0.6842, 0.9690, 0.001],
    [49,  0.6841, 0.9814, 0.6842, 0.9656, 0.001],
])

# Control Match Parameters
data2 = np.array([
    [0, 0.6935, 0.8363, 0.6850, 0.8861, 0.001],
    [1, 0.6848, 0.9159, 0.6847, 0.9128, 0.001],
    [2, 0.6847, 0.9266, 0.6846, 0.9222, 0.001],
    [3, 0.6846, 0.9327, 0.6845, 0.9342, 0.001],
    [4, 0.6845, 0.9387, 0.6845, 0.9387, 0.001],
    [5, 0.6845, 0.9410, 0.6845, 0.9393, 0.001],
    [6, 0.6845, 0.9434, 0.6844, 0.9404, 0.001],
    [7, 0.6844, 0.9443, 0.6844, 0.9456, 0.001],
    [8, 0.6844, 0.9467, 0.6844, 0.9485, 0.001],
    [9, 0.6844, 0.9476, 0.6844, 0.9431, 0.001],
    [10, 0.6844, 0.9483, 0.6843, 0.9531, 0.001],
    [11, 0.6844, 0.9490, 0.6843, 0.9530, 0.001],
    [12, 0.6844, 0.9499, 0.6844, 0.9508, 0.001],
    [13, 0.6844, 0.9510, 0.6843, 0.9531, 0.001],
    [14, 0.6844, 0.9523, 0.6843, 0.9561, 0.001],
    [15, 0.6844, 0.9531, 0.6843, 0.9558, 0.001],
    [16, 0.6844, 0.9531, 0.6844, 0.9495, 0.001],
    [17, 0.6844, 0.9535, 0.6844, 0.9469, 0.001],
    [18, 0.6843, 0.9542, 0.6843, 0.9549, 0.001],
    [19, 0.6843, 0.9547, 0.6844, 0.9487, 0.001],
    [20, 0.6843, 0.9552, 0.6843, 0.9518, 0.001],
    [21, 0.6843, 0.9560, 0.6843, 0.9572, 0.001],
    [22, 0.6843, 0.9563, 0.6844, 0.9499, 0.001],
    [23, 0.6843, 0.9560, 0.6843, 0.9526, 0.001],
    [24, 0.6843, 0.9567, 0.6843, 0.9564, 0.001],
    [25, 0.6843, 0.9572, 0.6844, 0.9515, 0.001],
    [26, 0.6843, 0.9575, 0.6843, 0.9527, 0.001],
    [27, 0.6843, 0.9582, 0.6843, 0.9598, 0.001],
    [28, 0.6843, 0.9584, 0.6843, 0.9550, 0.001],
    [29, 0.6843, 0.9594, 0.6843, 0.9557, 0.001],
    [30, 0.6843, 0.9590, 0.6843, 0.9575, 0.001],
    [31, 0.6843, 0.9594, 0.6843, 0.9597, 0.001],
    [32, 0.6843, 0.9603, 0.6843, 0.9585, 0.001],
    [33, 0.6843, 0.9602, 0.6843, 0.9535, 0.001],
    [34, 0.6843, 0.9599, 0.6843, 0.9531, 0.001],
    [35, 0.6843, 0.9599, 0.6843, 0.9558, 0.001],
    [36, 0.6843, 0.9611, 0.6843, 0.9559, 0.001],
    [37, 0.6843, 0.9609, 0.6843, 0.9556, 0.001],
    [38, 0.6843, 0.9602, 0.6843, 0.9583, 0.001],
    [39, 0.6843, 0.9607, 0.6843, 0.9574, 0.001],
    [40, 0.6843, 0.9615, 0.6843, 0.9567, 0.001],
    [41, 0.6843, 0.9618, 0.6843, 0.9594, 0.001],
    [42, 0.6843, 0.9620, 0.6843, 0.9572, 0.001],
    [43, 0.6843, 0.9620, 0.6843, 0.9618, 0.001],
    [44, 0.6843, 0.9617, 0.6843, 0.9562, 0.001],
    [45, 0.6843, 0.9620, 0.6843, 0.9557, 0.001],
    [46, 0.6843, 0.9622, 0.6843, 0.9592, 0.001],
    [47, 0.6843, 0.9626, 0.6843, 0.9580, 0.001],
    [48, 0.6843, 0.9632, 0.6843, 0.9608, 0.001],
    [49, 0.6843, 0.9633, 0.6843, 0.9622, 0.001],
])

# Current Divisive Inhibition
data3 = np.array([
    [0, 0.6991, 0.7828, 0.6853, 0.8630, 0.001],
    [1, 0.6854, 0.8650, 0.6850, 0.8869, 0.001],
    [2, 0.6850, 0.8884, 0.6848, 0.9097, 0.001],
    [3, 0.6847, 0.9192, 0.6846, 0.9283, 0.001],
    [4, 0.6846, 0.9332, 0.6845, 0.9327, 0.001],
    [5, 0.6845, 0.9372, 0.6845, 0.9359, 0.001],
    [6, 0.6845, 0.9441, 0.6844, 0.9469, 0.001],
    [7, 0.6844, 0.9473, 0.6844, 0.9512, 0.001],
    [8, 0.6844, 0.9511, 0.6844, 0.9530, 0.001],
    [9, 0.6844, 0.9524, 0.6844, 0.9547, 0.001],
    [10, 0.6843, 0.9551, 0.6844, 0.9528, 0.001],
    [11, 0.6843, 0.9576, 0.6844, 0.9526, 0.001],
    [12, 0.6843, 0.9586, 0.6843, 0.9610, 0.001],
    [13, 0.6843, 0.9608, 0.6843, 0.9594, 0.001],
    [14, 0.6843, 0.9620, 0.6843, 0.9613, 0.001],
    [15, 0.6843, 0.9613, 0.6843, 0.9582, 0.001],
    [16, 0.6843, 0.9635, 0.6843, 0.9577, 0.001],
    [17, 0.6842, 0.9644, 0.6843, 0.9595, 0.001],
    [18, 0.6842, 0.9654, 0.6843, 0.9608, 0.001],
    [19, 0.6842, 0.9666, 0.6843, 0.9607, 0.001],
    [20, 0.6842, 0.9665, 0.6843, 0.9587, 0.001],
    [21, 0.6842, 0.9663, 0.6843, 0.9626, 0.001],
    [22, 0.6842, 0.9677, 0.6843, 0.9634, 0.001],
    [23, 0.6842, 0.9688, 0.6843, 0.9640, 0.001],
    [24, 0.6842, 0.9684, 0.6843, 0.9621, 0.001],
    [25, 0.6842, 0.9691, 0.6843, 0.9639, 0.001],
    [26, 0.6842, 0.9700, 0.6842, 0.9641, 0.001],
    [27, 0.6842, 0.9710, 0.6842, 0.9645, 0.001],
    [28, 0.6842, 0.9710, 0.6842, 0.9665, 0.001],
    [29, 0.6842, 0.9717, 0.6842, 0.9665, 0.001],
    [30, 0.6842, 0.9725, 0.6842, 0.9660, 0.001],
    [31, 0.6842, 0.9727, 0.6842, 0.9650, 0.001],
    [32, 0.6842, 0.9731, 0.6842, 0.9649, 0.001],
    [33, 0.6842, 0.9735, 0.6842, 0.9645, 0.001],
    [34, 0.6841, 0.9743, 0.6843, 0.9666, 0.001],
    [35, 0.6841, 0.9748, 0.6842, 0.9693, 0.001],
    [36, 0.6841, 0.9752, 0.6842, 0.9642, 0.001],
    [37, 0.6841, 0.9747, 0.6842, 0.9681, 0.001],
    [38, 0.6841, 0.9755, 0.6842, 0.9695, 0.001],
    [39, 0.6841, 0.9760, 0.6842, 0.9666, 0.001],
    [40, 0.6841, 0.9760, 0.6842, 0.9676, 0.001],
    [41, 0.6841, 0.9770, 0.6842, 0.9663, 0.001],
    [42, 0.6841, 0.9762, 0.6842, 0.9687, 0.001],
    [43, 0.6841, 0.9769, 0.6842, 0.9662, 0.001],
    [44, 0.6843, 0.9569, 0.6844, 0.9533, 0.001],
    [45, 0.6843, 0.9639, 0.6843, 0.9604, 0.001],
    [46, 0.6842, 0.9696, 0.6843, 0.9638, 0.001],
    [47, 0.6842, 0.9730, 0.6842, 0.9651, 0.001],
    [48, 0.6841, 0.9753, 0.6842, 0.9657, 0.001],
    [49, 0.6841, 0.9775, 0.6842, 0.9691, 0.001],
])

# Control Match Iterations and Parameters
data4 = np.array([
    [0, 0.7049, 0.8303, 0.6853, 0.8520, 0.001],
    [1, 0.6849, 0.9137, 0.6852, 0.8512, 0.001],
    [2, 0.6847, 0.9268, 0.6850, 0.8735, 0.001],
    [3, 0.6846, 0.9345, 0.6847, 0.9144, 0.001],
    [4, 0.6845, 0.9387, 0.6847, 0.9069, 0.001],
    [5, 0.6845, 0.9423, 0.6850, 0.8754, 0.001],
    [6, 0.6844, 0.9446, 0.6848, 0.8953, 0.001],
    [7, 0.6844, 0.9470, 0.6847, 0.9061, 0.001],
    [8, 0.6844, 0.9486, 0.6847, 0.9117, 0.001],
    [9, 0.6844, 0.9505, 0.6847, 0.9102, 0.001],
    [10, 0.6844, 0.9523, 0.6845, 0.9262, 0.001],
    [11, 0.6844, 0.9530, 0.6845, 0.9272, 0.001],
    [12, 0.6843, 0.9547, 0.6845, 0.9322, 0.001],
    [13, 0.6843, 0.9557, 0.6845, 0.9310, 0.001],
    [14, 0.6843, 0.9569, 0.6845, 0.9307, 0.001],
    [15, 0.6843, 0.9569, 0.6846, 0.9152, 0.001],
    [16, 0.6843, 0.9572, 0.6845, 0.9284, 0.001],
    [17, 0.6843, 0.9584, 0.6845, 0.9272, 0.001],
    [18, 0.6843, 0.9599, 0.6844, 0.9441, 0.001],
    [19, 0.6843, 0.9599, 0.6844, 0.9354, 0.001],
    [20, 0.6843, 0.9605, 0.6845, 0.9338, 0.001],
    [21, 0.6843, 0.9606, 0.6844, 0.9422, 0.001],
    [22, 0.6843, 0.9612, 0.6844, 0.9363, 0.001],
    [23, 0.6843, 0.9622, 0.6845, 0.9307, 0.001],
    [24, 0.6843, 0.9621, 0.6844, 0.9363, 0.001],
    [25, 0.6843, 0.9638, 0.6843, 0.9532, 0.001],
    [26, 0.6843, 0.9633, 0.6844, 0.9499, 0.001],
    [27, 0.6843, 0.9627, 0.6844, 0.9440, 0.001],
    [28, 0.6843, 0.9638, 0.6843, 0.9582, 0.001],
    [29, 0.6842, 0.9645, 0.6844, 0.9499, 0.001],
    [30, 0.6842, 0.9653, 0.6843, 0.9548, 0.001],
    [31, 0.6842, 0.9652, 0.6844, 0.9513, 0.001],
    [32, 0.6842, 0.9656, 0.6844, 0.9503, 0.001],
    [33, 0.6842, 0.9653, 0.6844, 0.9513, 0.001],
    [34, 0.6842, 0.9664, 0.6843, 0.9586, 0.001],
    [35, 0.6842, 0.9663, 0.6845, 0.9297, 0.001],
    [36, 0.6842, 0.9667, 0.6844, 0.9455, 0.001],
    [37, 0.6842, 0.9672, 0.6844, 0.9514, 0.001],
    [38, 0.6842, 0.9666, 0.6844, 0.9383, 0.001],
    [39, 0.6842, 0.9677, 0.6844, 0.9542, 0.001],
    [40, 0.6842, 0.9679, 0.6844, 0.9475, 0.001],
    [41, 0.6842, 0.9672, 0.6843, 0.9548, 0.001],
    [42, 0.6842, 0.9682, 0.6843, 0.9555, 0.001],
    [43, 0.6842, 0.9674, 0.6844, 0.9430, 0.001],
    [44, 0.6842, 0.9685, 0.6845, 0.9395, 0.001],
    [45, 0.6842, 0.9684, 0.6844, 0.9450, 0.001],
    [46, 0.6842, 0.9692, 0.6844, 0.9521, 0.001],
    [47, 0.6842, 0.9697, 0.6843, 0.9615, 0.001],
    [48, 0.6842, 0.9685, 0.6845, 0.9500, 0.001],
    [49, 0.6842, 0.9684, 0.6843, 0.9542, 0.001],
])

# Control model Classification head only
data5 = np.array([
    [0, 0.6944, 0.6003, 0.6870, 0.7103, 0.001],
    [1, 0.6866, 0.7222, 0.6865, 0.7284, 0.001],
    [2, 0.6864, 0.7357, 0.6863, 0.7454, 0.001],
    [3, 0.6863, 0.7440, 0.6862, 0.7477, 0.001],
    [4, 0.6863, 0.7470, 0.6863, 0.7488, 0.001],
    [5, 0.6863, 0.7470, 0.6863, 0.7504, 0.001],
    [6, 0.6863, 0.7490, 0.6862, 0.7523, 0.001],
    [7, 0.6863, 0.7490, 0.6862, 0.7514, 0.001],
    [8, 0.6862, 0.7503, 0.6862, 0.7540, 0.001],
    [9, 0.6862, 0.7507, 0.6862, 0.7536, 0.001],
    [10, 0.6862, 0.7524, 0.6862, 0.7532, 0.001],
    [11, 0.6862, 0.7522, 0.6862, 0.7522, 0.001],
    [12, 0.6862, 0.7535, 0.6862, 0.7551, 0.001],
    [13, 0.6862, 0.7533, 0.6862, 0.7535, 0.001],
    [14, 0.6862, 0.7531, 0.6862, 0.7509, 0.001],
    [15, 0.6862, 0.7543, 0.6862, 0.7547, 0.001],
    [16, 0.6862, 0.7549, 0.6862, 0.7521, 0.001],
    [17, 0.6862, 0.7545, 0.6862, 0.7494, 0.001],
    [18, 0.6862, 0.7553, 0.6862, 0.7561, 0.001],
    [19, 0.6862, 0.7549, 0.6862, 0.7547, 0.001],
    [20, 0.6862, 0.7559, 0.6862, 0.7564, 0.001],
    [21, 0.6862, 0.7556, 0.6862, 0.7534, 0.001],
    [22, 0.6862, 0.7557, 0.6862, 0.7559, 0.001],
    [23, 0.6862, 0.7557, 0.6861, 0.7572, 0.001],
    [24, 0.6862, 0.7568, 0.6861, 0.7568, 0.001],
    [25, 0.6862, 0.7558, 0.6861, 0.7576, 0.001],
    [26, 0.6862, 0.7571, 0.6861, 0.7575, 0.001],
    [27, 0.6862, 0.7564, 0.6862, 0.7561, 0.001],
    [28, 0.6862, 0.7572, 0.6861, 0.7567, 0.001],
    [29, 0.6862, 0.7559, 0.6861, 0.7579, 0.001],
    [30, 0.6862, 0.7573, 0.6861, 0.7571, 0.001],
    [31, 0.6862, 0.7571, 0.6862, 0.7572, 0.001],
    [32, 0.6862, 0.7574, 0.6862, 0.7562, 0.001],
    [33, 0.6862, 0.7564, 0.6861, 0.7589, 0.001],
    [34, 0.6862, 0.7573, 0.6862, 0.7566, 0.001],
    [35, 0.6862, 0.7576, 0.6861, 0.7569, 0.001],
    [36, 0.6862, 0.7572, 0.6861, 0.7567, 0.001],
    [37, 0.6862, 0.7579, 0.6862, 0.7571, 0.001],
    [38, 0.6862, 0.7576, 0.6861, 0.7576, 0.001],
    [39, 0.6862, 0.7576, 0.6861, 0.7587, 0.001],
    [40, 0.6862, 0.7574, 0.6861, 0.7580, 0.001],
    [41, 0.6862, 0.7576, 0.6861, 0.7573, 0.001],
    [42, 0.6862, 0.7578, 0.6861, 0.7575, 0.001],
    [43, 0.6862, 0.7580, 0.6862, 0.7563, 0.001],
    [44, 0.6862, 0.7578, 0.6861, 0.7580, 0.001],
    [45, 0.6862, 0.7586, 0.6861, 0.7584, 0.001],
    [46, 0.6862, 0.7583, 0.6861, 0.7587, 0.001],
    [47, 0.6862, 0.7579, 0.6861, 0.7579, 0.001],
    [48, 0.6862, 0.7583, 0.6861, 0.7586, 0.001],
    [49, 0.6862, 0.7586, 0.6861, 0.7582, 0.001],
])

plt.ion()

# loss figure
plt.figure('Loss')
plt.plot(data1[:, 0], data1[:, 3], label='current_subtract_val')
plt.plot(data2[:, 0], data2[:, 3], label='control_match_param_val')
plt.plot(data3[:, 0], data3[:, 3], label='current_divisive_val')
plt.plot(data4[:, 0], data4[:, 3], label='control_match_iterations_val')
plt.plot(data5[:, 0], data5[:, 3], label='control_class_head_only_val')
plt.plot(data1[:, 0], data1[:, 1], linestyle=':', label='current_subtract_train')
plt.plot(data2[:, 0], data2[:, 1], linestyle=':', label='control_match_param_train')
plt.plot(data3[:, 0], data3[:, 1], linestyle=':', label='current_divisive__train')
plt.plot(data4[:, 0], data4[:, 1], linestyle=':', label='control_match_iterations_train')
plt.plot(data5[:, 0], data5[:, 1], linestyle=':', label='control_class_head_only_train')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

# plt.figure('IoU')
f, ax_arr = plt.subplots(2, 1, sharex=True)
ax_arr[0].plot(data1[:, 0], data1[:, 4], label='current_subtract_val')
ax_arr[0].plot(data2[:, 0], data2[:, 4], label='control_match_param_val')
ax_arr[0].plot(data3[:, 0], data3[:, 4], label='current_divisive_val')
ax_arr[0].plot(data4[:, 0], data4[:, 4], label='control_match_iterations_val')
ax_arr[0].plot(data5[:, 0], data5[:, 4], label='control_class_head_only_val')
ax_arr[1].plot(data1[:, 0], data1[:, 2], linestyle=':', label='current_subtract_train')
ax_arr[1].plot(data2[:, 0], data2[:, 2], linestyle=':', label='control_match_param_train')
ax_arr[1].plot(data3[:, 0], data3[:, 2], linestyle=':', label='current_divisive_train')
ax_arr[1].plot(data4[:, 0], data4[:, 2], linestyle=':', label='control_match_iterations_train')
ax_arr[1].plot(data5[:, 0], data5[:, 2], linestyle=':', label='control_class_head_only_train')

ax_arr[0].set_ylabel("IoU")
ax_arr[0].grid()
ax_arr[0].legend()
ax_arr[1].set_ylabel("IoU")
ax_arr[1].set_xlabel("Epoch")
ax_arr[1].grid()
ax_arr[1].legend()
plt.tight_layout()

input("Press any key to continue")
