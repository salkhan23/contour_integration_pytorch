# ---------------------------------------------------------------------------------------
# Plot results for the puncture bubbles experiment - explore num bubbles
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import plot_puncture_bubble_size_results as plotting_fcns

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)

results_n_bubbles_0 = {
    'iou_vs_epoch': np.array([
        [1, 0.2416, 0.2276, 0.1265, 0.4288, 0.001],
        [2, 0.1185, 0.4621, 0.1102, 0.4902, 0.001],
        [3, 0.1094, 0.4917, 0.1067, 0.5021, 0.001],
        [4, 0.1072, 0.5006, 0.1047, 0.5066, 0.001],
        [5, 0.1051, 0.5083, 0.1040, 0.5229, 0.001],
        [6, 0.1042, 0.5133, 0.1026, 0.5324, 0.001],
        [7, 0.1067, 0.5132, 0.1077, 0.5227, 0.001],
        [8, 0.1036, 0.5208, 0.1009, 0.5386, 0.001],
        [9, 0.1024, 0.5231, 0.0998, 0.5384, 0.001],
        [10, 0.1013, 0.5267, 0.0989, 0.5393, 0.001],
        [11, 0.1008, 0.5289, 0.0986, 0.5424, 0.001],
        [12, 0.1008, 0.5311, 0.0982, 0.5434, 0.001],
        [13, 0.1003, 0.5324, 0.0982, 0.5423, 0.001],
        [14, 0.0997, 0.5341, 0.0984, 0.5470, 0.001],
        [15, 0.0994, 0.5355, 0.0986, 0.5490, 0.001],
        [16, 0.0992, 0.5367, 0.0973, 0.5487, 0.001],
        [17, 0.0991, 0.5375, 0.0973, 0.5499, 0.001],
        [18, 0.0990, 0.5386, 0.0976, 0.5518, 0.001],
        [19, 0.0985, 0.5395, 0.0975, 0.5535, 0.001],
        [20, 0.0983, 0.5406, 0.0974, 0.5538, 0.001],
        [21, 0.0981, 0.5418, 0.0971, 0.5558, 0.001],
        [22, 0.0979, 0.5425, 0.0970, 0.5566, 0.001],
        [23, 0.0977, 0.5432, 0.0974, 0.5572, 0.001],
        [24, 0.0976, 0.5437, 0.0971, 0.5570, 0.001],
        [25, 0.0976, 0.5441, 0.0976, 0.5568, 0.001],
        [26, 0.0975, 0.5446, 0.0970, 0.5567, 0.001],
        [27, 0.0972, 0.5454, 0.0967, 0.5573, 0.001],
        [28, 0.0972, 0.5457, 0.0970, 0.5572, 0.001],
        [29, 0.0971, 0.5461, 0.0970, 0.5566, 0.001],
        [30, 0.0971, 0.5464, 0.0965, 0.5586, 0.001],
        [31, 0.0970, 0.5467, 0.0965, 0.5590, 0.001],
        [32, 0.0882, 0.5589, 0.0870, 0.5623, 0.0001],
        [33, 0.0879, 0.5598, 0.0869, 0.5629, 0.0001],
        [34, 0.0878, 0.5603, 0.0867, 0.5634, 0.0001],
        [35, 0.0877, 0.5607, 0.0866, 0.5640, 0.0001],
        [36, 0.0876, 0.5610, 0.0866, 0.5645, 0.0001],
        [37, 0.0876, 0.5612, 0.0865, 0.5649, 0.0001],
        [38, 0.0875, 0.5615, 0.0864, 0.5653, 0.0001],
        [39, 0.0874, 0.5617, 0.0864, 0.5656, 0.0001],
        [40, 0.0874, 0.5619, 0.0863, 0.5662, 0.0001],
        [41, 0.0873, 0.5621, 0.0863, 0.5666, 0.0001],
        [42, 0.0873, 0.5623, 0.0862, 0.5668, 0.0001],
        [43, 0.0872, 0.5624, 0.0862, 0.5671, 0.0001],
        [44, 0.0872, 0.5626, 0.0861, 0.5673, 0.0001],
        [45, 0.0872, 0.5627, 0.0861, 0.5676, 0.0001],
        [46, 0.0871, 0.5629, 0.0860, 0.5678, 0.0001],
        [47, 0.0871, 0.5630, 0.0860, 0.5681, 0.0001],
        [48, 0.0870, 0.5631, 0.0860, 0.5683, 0.0001],
        [49, 0.0870, 0.5633, 0.0859, 0.5686, 0.0001],
        [50, 0.0870, 0.5634, 0.0859, 0.5689, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0016, 1.0056, 1.0077, 1.0089]),
        'std_gain': np.array([0.0638, 0.0595, 0.0623, 0.0643, 0.0636])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0130, 1.0197, 1.0240, 1.0248]),
        'std_gain': np.array([0.0431, 0.0348, 0.0307, 0.0295, 0.0323])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0058, 1.0174, 1.0359, 1.0416, 1.0446, 1.0473, 1.0521, 1.0514]),
        'std_gain': np.array([0.0617, 0.0536, 0.0581, 0.0470, 0.0440, 0.0427, 0.0419, 0.0346])
    },
    'no_optimal_stimuli_neurons': [13, 31, 34, 43, 44, 52, 62],
    'filtered_out_neurons': [22, 51]
}

results_n_bubbles_50 = {
    'iou_vs_epoch': np.array([
        [1, 0.2333, 0.0825, 0.1436, 0.3636, 0.001],
        [2, 0.1299, 0.4172, 0.1290, 0.4151, 0.001],
        [3, 0.1233, 0.4415, 0.1212, 0.4336, 0.001],
        [4, 0.1198, 0.4537, 0.1191, 0.4403, 0.001],
        [5, 0.1181, 0.4607, 0.1161, 0.4528, 0.001],
        [6, 0.1169, 0.4660, 0.1145, 0.4676, 0.001],
        [7, 0.1161, 0.4695, 0.1148, 0.4796, 0.001],
        [8, 0.1157, 0.4727, 0.1140, 0.4848, 0.001],
        [9, 0.1149, 0.4752, 0.1142, 0.4811, 0.001],
        [10, 0.1143, 0.4771, 0.1135, 0.4825, 0.001],
        [11, 0.1140, 0.4785, 0.1128, 0.4866, 0.001],
        [12, 0.1137, 0.4800, 0.1127, 0.4867, 0.001],
        [13, 0.1135, 0.4813, 0.1129, 0.4868, 0.001],
        [14, 0.1134, 0.4824, 0.1124, 0.4849, 0.001],
        [15, 0.1130, 0.4834, 0.1128, 0.4747, 0.001],
        [16, 0.1128, 0.4843, 0.1127, 0.4751, 0.001],
        [17, 0.1124, 0.4855, 0.1127, 0.4771, 0.001],
        [18, 0.1122, 0.4864, 0.1127, 0.4754, 0.001],
        [19, 0.1121, 0.4873, 0.1123, 0.4786, 0.001],
        [20, 0.1124, 0.4881, 0.1130, 0.4747, 0.001],
        [21, 0.1121, 0.4883, 0.1130, 0.4746, 0.001],
        [22, 0.1116, 0.4894, 0.1127, 0.4787, 0.001],
        [23, 0.1115, 0.4900, 0.1126, 0.4771, 0.001],
        [24, 0.1113, 0.4907, 0.1127, 0.4766, 0.001],
        [25, 0.1113, 0.4912, 0.1128, 0.4774, 0.001],
        [26, 0.1112, 0.4916, 0.1123, 0.4817, 0.001],
        [27, 0.1110, 0.4922, 0.1121, 0.4867, 0.001],
        [28, 0.1110, 0.4927, 0.1120, 0.4867, 0.001],
        [29, 0.1109, 0.4930, 0.1117, 0.4866, 0.001],
        [30, 0.1107, 0.4936, 0.1118, 0.4878, 0.001],
        [31, 0.1111, 0.4940, 0.1116, 0.4891, 0.001],
        [32, 0.1010, 0.5077, 0.1003, 0.5177, 0.0001],
        [33, 0.1006, 0.5091, 0.1003, 0.5190, 0.0001],
        [34, 0.1005, 0.5097, 0.1002, 0.5193, 0.0001],
        [35, 0.1003, 0.5102, 0.1001, 0.5196, 0.0001],
        [36, 0.1002, 0.5106, 0.1001, 0.5198, 0.0001],
        [37, 0.1001, 0.5110, 0.1000, 0.5199, 0.0001],
        [38, 0.1000, 0.5114, 0.0999, 0.5201, 0.0001],
        [39, 0.1000, 0.5117, 0.0998, 0.5202, 0.0001],
        [40, 0.0999, 0.5120, 0.0998, 0.5204, 0.0001],
        [41, 0.0998, 0.5123, 0.0997, 0.5203, 0.0001],
        [42, 0.0997, 0.5125, 0.0997, 0.5204, 0.0001],
        [43, 0.0997, 0.5128, 0.0996, 0.5206, 0.0001],
        [44, 0.0996, 0.5130, 0.0996, 0.5207, 0.0001],
        [45, 0.0996, 0.5132, 0.0995, 0.5207, 0.0001],
        [46, 0.0995, 0.5134, 0.0995, 0.5207, 0.0001],
        [47, 0.0995, 0.5136, 0.0995, 0.5208, 0.0001],
        [48, 0.0994, 0.5139, 0.0994, 0.5209, 0.0001],
        [49, 0.0994, 0.5141, 0.0994, 0.5211, 0.0001],
        [50, 0.0993, 0.5143, 0.0994, 0.5210, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 0.9960, 1.0015, 0.9950, 0.9973]),
        'std_gain': np.array([0.1347, 0.1365, 0.1241, 0.1111, 0.1046])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0494, 1.0646, 1.0666, 1.0683]),
        'std_gain': np.array([0.0773, 0.0658, 0.0687, 0.0721, 0.0697])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0011, 1.0187, 1.0548, 1.0746, 1.0970, 1.1100, 1.1253, 1.1308]),
        'std_gain': np.array([0.0992, 0.0914, 0.0884, 0.0848, 0.0780, 0.0736, 0.0728, 0.0646])
    },
    'no_optimal_stimuli_neurons': [25, 34, 42, 44, 62],
    'filtered_out_neurons': [13, 14, 18, 23, 28, 31, 48, 54]
}

results_n_bubbles_100 = {
    'iou_vs_epoch': np.array([
        [1, 0.2852, 0.1213, 0.1473, 0.3524, 0.001],
        [2, 0.1414, 0.3749, 0.1391, 0.3894, 0.001],
        [3, 0.1345, 0.3946, 0.1315, 0.3939, 0.001],
        [4, 0.1303, 0.4097, 0.1294, 0.3920, 0.001],
        [5, 0.1283, 0.4196, 0.1301, 0.4036, 0.001],
        [6, 0.1260, 0.4272, 0.1261, 0.4316, 0.001],
        [7, 0.1249, 0.4328, 0.1252, 0.4326, 0.001],
        [8, 0.1236, 0.4373, 0.1236, 0.4394, 0.001],
        [9, 0.1239, 0.4406, 0.1231, 0.4405, 0.001],
        [10, 0.1227, 0.4437, 0.1240, 0.4381, 0.001],
        [11, 0.1222, 0.4462, 0.1226, 0.4391, 0.001],
        [12, 0.1213, 0.4488, 0.1225, 0.4397, 0.001],
        [13, 0.1211, 0.4507, 0.1232, 0.4339, 0.001],
        [14, 0.1211, 0.4525, 0.1213, 0.4389, 0.001],
        [15, 0.1207, 0.4541, 0.1204, 0.4444, 0.001],
        [16, 0.1200, 0.4555, 0.1199, 0.4457, 0.001],
        [17, 0.1200, 0.4565, 0.1202, 0.4407, 0.001],
        [18, 0.1198, 0.4577, 0.1191, 0.4506, 0.001],
        [19, 0.1194, 0.4588, 0.1180, 0.4587, 0.001],
        [20, 0.1192, 0.4596, 0.1180, 0.4522, 0.001],
        [21, 0.1190, 0.4603, 0.1179, 0.4520, 0.001],
        [22, 0.1191, 0.4609, 0.1180, 0.4499, 0.001],
        [23, 0.1188, 0.4615, 0.1184, 0.4517, 0.001],
        [24, 0.1188, 0.4620, 0.1175, 0.4548, 0.001],
        [25, 0.1182, 0.4635, 0.1176, 0.4570, 0.001],
        [26, 0.1182, 0.4639, 0.1170, 0.4576, 0.001],
        [27, 0.1183, 0.4646, 0.1167, 0.4583, 0.001],
        [28, 0.1180, 0.4649, 0.1174, 0.4565, 0.001],
        [29, 0.1178, 0.4655, 0.1165, 0.4587, 0.001],
        [30, 0.1179, 0.4660, 0.1168, 0.4588, 0.001],
        [31, 0.1180, 0.4665, 0.1162, 0.4602, 0.001],
        [32, 0.1073, 0.4811, 0.1062, 0.4830, 0.0001],
        [33, 0.1069, 0.4826, 0.1060, 0.4836, 0.0001],
        [34, 0.1068, 0.4834, 0.1059, 0.4840, 0.0001],
        [35, 0.1066, 0.4840, 0.1058, 0.4845, 0.0001],
        [36, 0.1065, 0.4845, 0.1057, 0.4846, 0.0001],
        [37, 0.1064, 0.4850, 0.1056, 0.4849, 0.0001],
        [38, 0.1063, 0.4854, 0.1055, 0.4851, 0.0001],
        [39, 0.1062, 0.4857, 0.1055, 0.4852, 0.0001],
        [40, 0.1061, 0.4861, 0.1054, 0.4853, 0.0001],
        [41, 0.1060, 0.4864, 0.1053, 0.4857, 0.0001],
        [42, 0.1059, 0.4867, 0.1053, 0.4858, 0.0001],
        [43, 0.1059, 0.4870, 0.1052, 0.4859, 0.0001],
        [44, 0.1058, 0.4872, 0.1052, 0.4859, 0.0001],
        [45, 0.1057, 0.4875, 0.1051, 0.4858, 0.0001],
        [46, 0.1057, 0.4878, 0.1051, 0.4865, 0.0001],
        [47, 0.1056, 0.4880, 0.1050, 0.4864, 0.0001],
        [48, 0.1056, 0.4883, 0.1050, 0.4866, 0.0001],
        [49, 0.1055, 0.4885, 0.1049, 0.4869, 0.0001],
        [50, 0.1054, 0.4888, 0.1049, 0.4870, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 0.9997, 1.0055, 1.0097, 1.0001]),
        'std_gain': np.array([0.1347, 0.1365, 0.1241, 0.1111, 0.1046])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0282, 1.0627, 1.0697, 1.0713]),
        'std_gain': np.array([0.0906, 0.0930, 0.0997, 0.0850, 0.1013])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([0.9944, 1.0618, 1.1390, 1.1979, 1.2264, 1.2644, 1.2790, 1.2962]),
        'std_gain': np.array([0.1515, 0.1589, 0.1305, 0.1217, 0.1097, 0.1195, 0.1063, 0.1100])
    },
    'no_optimal_stimuli_neurons': [2, 5, 19, 23, 25, 44, 59],
    'filtered_out_neurons': [1, 17, 41]
}

results_n_bubbles_200 = {
    'iou_vs_epoch': np.array([
        [1, 0.3067, 0.0888, 0.1803, 0.1980, 0.001],
        [2, 0.1626, 0.2743, 0.1553, 0.3123, 0.001],
        [3, 0.1524, 0.3247, 0.1491, 0.3302, 0.001],
        [4, 0.1467, 0.3423, 0.1451, 0.3336, 0.001],
        [5, 0.1441, 0.3523, 0.1431, 0.3434, 0.001],
        [6, 0.1426, 0.3594, 0.1418, 0.3550, 0.001],
        [7, 0.1405, 0.3661, 0.1394, 0.3555, 0.001],
        [8, 0.1406, 0.3708, 0.1403, 0.3559, 0.001],
        [9, 0.1395, 0.3752, 0.1392, 0.3599, 0.001],
        [10, 0.1369, 0.3796, 0.1379, 0.3612, 0.001],
        [11, 0.1365, 0.3824, 0.1379, 0.3626, 0.001],
        [12, 0.1360, 0.3850, 0.1382, 0.3703, 0.001],
        [13, 0.1352, 0.3872, 0.1374, 0.3744, 0.001],
        [14, 0.1348, 0.3896, 0.1377, 0.3806, 0.001],
        [15, 0.1351, 0.3911, 0.1370, 0.3854, 0.001],
        [16, 0.1344, 0.3932, 0.1368, 0.3888, 0.001],
        [17, 0.1346, 0.3939, 0.1360, 0.3953, 0.001],
        [18, 0.1333, 0.3965, 0.1362, 0.4014, 0.001],
        [19, 0.1331, 0.3979, 0.1367, 0.4037, 0.001],
        [20, 0.1330, 0.3992, 0.1362, 0.4033, 0.001],
        [21, 0.1327, 0.4003, 0.1352, 0.4051, 0.001],
        [22, 0.1324, 0.4014, 0.1345, 0.4052, 0.001],
        [23, 0.1322, 0.4024, 0.1342, 0.4114, 0.001],
        [24, 0.1323, 0.4032, 0.1337, 0.4062, 0.001],
        [25, 0.1318, 0.4040, 0.1337, 0.4103, 0.001],
        [26, 0.1316, 0.4048, 0.1338, 0.4114, 0.001],
        [27, 0.1316, 0.4054, 0.1342, 0.4118, 0.001],
        [28, 0.1314, 0.4062, 0.1337, 0.4156, 0.001],
        [29, 0.1314, 0.4069, 0.1337, 0.4090, 0.001],
        [30, 0.1311, 0.4074, 0.1333, 0.4113, 0.001],
        [31, 0.1310, 0.4082, 0.1340, 0.4093, 0.001],
        [32, 0.1206, 0.4240, 0.1194, 0.4179, 0.0001],
        [33, 0.1199, 0.4256, 0.1193, 0.4172, 0.0001],
        [34, 0.1198, 0.4262, 0.1192, 0.4173, 0.0001],
        [35, 0.1196, 0.4267, 0.1190, 0.4177, 0.0001],
        [36, 0.1195, 0.4272, 0.1189, 0.4176, 0.0001],
        [37, 0.1194, 0.4276, 0.1188, 0.4179, 0.0001],
        [38, 0.1193, 0.4279, 0.1187, 0.4182, 0.0001],
        [39, 0.1192, 0.4283, 0.1187, 0.4185, 0.0001],
        [40, 0.1191, 0.4286, 0.1185, 0.4190, 0.0001],
        [41, 0.1191, 0.4290, 0.1185, 0.4184, 0.0001],
        [42, 0.1190, 0.4293, 0.1185, 0.4187, 0.0001],
        [43, 0.1189, 0.4296, 0.1184, 0.4187, 0.0001],
        [44, 0.1188, 0.4300, 0.1184, 0.4184, 0.0001],
        [45, 0.1188, 0.4303, 0.1183, 0.4187, 0.0001],
        [46, 0.1187, 0.4305, 0.1182, 0.4189, 0.0001],
        [47, 0.1186, 0.4308, 0.1182, 0.4188, 0.0001],
        [48, 0.1185, 0.4311, 0.1182, 0.4188, 0.0001],
        [49, 0.1185, 0.4313, 0.1181, 0.4189, 0.0001],
        [50, 0.1184, 0.4316, 0.1180, 0.4196, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0421, 1.0456, 1.0439, 1.0423]),
        'std_gain': np.array([0.1365, 0.1215, 0.1194, 0.1184, 0.1203])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0899, 1.0929, 1.0948, 1.0916]),
        'std_gain': np.array([0.2241, 0.1275, 0.1024, 0.1147, 0.0961])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0432, 1.0656, 1.0848, 1.0986, 1.1089, 1.1191, 1.1261, 1.1269]),
        'std_gain': np.array([0.1257, 0.1054, 0.0879, 0.0861, 0.0808, 0.0747, 0.0657, 0.0591])
    },
    'no_optimal_stimuli_neurons': [1, 5, 13, 21, 23, 29, 34, 42, 44, 45, 48, 57, 58],
    'filtered_out_neurons': [9, 20, 41, 50]
}

results_n_bubbles_300 = {
    'iou_vs_epoch': np.array([
        [1, 0.4231, 0.0480, 0.2091, 0.1297, 0.001],
        [2, 0.1889, 0.1962, 0.1812, 0.2464, 0.001],
        [3, 0.1665, 0.2739, 0.1617, 0.3071, 0.001],
        [4, 0.1563, 0.3102, 0.1570, 0.3412, 0.001],
        [5, 0.1524, 0.3256, 0.1526, 0.3483, 0.001],
        [6, 0.1503, 0.3337, 0.1502, 0.3589, 0.001],
        [7, 0.1480, 0.3395, 0.1479, 0.3577, 0.001],
        [8, 0.1466, 0.3437, 0.1460, 0.3632, 0.001],
        [9, 0.1463, 0.3462, 0.1450, 0.3676, 0.001],
        [10, 0.1451, 0.3491, 0.1445, 0.3725, 0.001],
        [11, 0.1447, 0.3516, 0.1443, 0.3742, 0.001],
        [12, 0.1440, 0.3538, 0.1437, 0.3731, 0.001],
        [13, 0.1435, 0.3555, 0.1439, 0.3714, 0.001],
        [14, 0.1431, 0.3573, 0.1427, 0.3702, 0.001],
        [15, 0.1429, 0.3586, 0.1430, 0.3707, 0.001],
        [16, 0.1425, 0.3603, 0.1425, 0.3690, 0.001],
        [17, 0.1422, 0.3614, 0.1432, 0.3710, 0.001],
        [18, 0.1418, 0.3625, 0.1417, 0.3767, 0.001],
        [19, 0.1418, 0.3633, 0.1411, 0.3825, 0.001],
        [20, 0.1414, 0.3641, 0.1408, 0.3795, 0.001],
        [21, 0.1411, 0.3653, 0.1406, 0.3790, 0.001],
        [22, 0.1410, 0.3658, 0.1398, 0.3849, 0.001],
        [23, 0.1407, 0.3669, 0.1399, 0.3802, 0.001],
        [24, 0.1422, 0.3666, 0.1399, 0.3877, 0.001],
        [25, 0.1405, 0.3682, 0.1398, 0.3886, 0.001],
        [26, 0.1404, 0.3690, 0.1394, 0.3874, 0.001],
        [27, 0.1402, 0.3696, 0.1410, 0.3876, 0.001],
        [28, 0.1400, 0.3705, 0.1402, 0.3933, 0.001],
        [29, 0.1400, 0.3710, 0.1398, 0.3865, 0.001],
        [30, 0.1400, 0.3719, 0.1400, 0.3939, 0.001],
        [31, 0.1395, 0.3725, 0.1398, 0.3937, 0.001],
        [32, 0.1286, 0.3879, 0.1278, 0.3939, 0.0001],
        [33, 0.1281, 0.3896, 0.1280, 0.3981, 0.0001],
        [34, 0.1279, 0.3905, 0.1280, 0.4002, 0.0001],
        [35, 0.1278, 0.3912, 0.1279, 0.4011, 0.0001],
        [36, 0.1276, 0.3917, 0.1278, 0.4017, 0.0001],
        [37, 0.1275, 0.3922, 0.1278, 0.4027, 0.0001],
        [38, 0.1274, 0.3927, 0.1277, 0.4033, 0.0001],
        [39, 0.1273, 0.3930, 0.1276, 0.4038, 0.0001],
        [40, 0.1272, 0.3934, 0.1275, 0.4045, 0.0001],
        [41, 0.1271, 0.3938, 0.1274, 0.4048, 0.0001],
        [42, 0.1270, 0.3941, 0.1273, 0.4051, 0.0001],
        [43, 0.1270, 0.3944, 0.1272, 0.4052, 0.0001],
        [44, 0.1269, 0.3947, 0.1271, 0.4057, 0.0001],
        [45, 0.1268, 0.3950, 0.1271, 0.4061, 0.0001],
        [46, 0.1268, 0.3953, 0.1270, 0.4063, 0.0001],
        [47, 0.1267, 0.3956, 0.1269, 0.4062, 0.0001],
        [48, 0.1266, 0.3958, 0.1268, 0.4064, 0.0001],
        [49, 0.1266, 0.3961, 0.1267, 0.4066, 0.0001],
        [50, 0.1265, 0.3963, 0.1266, 0.4068, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.1178, 1.1283, 1.1316, 1.1305]),
        'std_gain': np.array([0.3977, 0.4832, 0.5210, 0.5128, 0.5126])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0852, 1.0941, 1.0892, 1.0982]),
        'std_gain': np.array([0.1423, 0.1474, 0.1674, 0.1563, 0.1577])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.1531, 1.3128, 1.4875, 1.5510, 1.7025, 1.7504, 1.8570, 1.9127]),
        'std_gain': np.array([0.5769, 0.5906, 0.6342, 0.6188, 0.4294, 0.3767, 0.3130, 0.2874])
    },
    'no_optimal_stimuli_neurons': [1, 3, 4, 7, 8, 12, 13, 14, 29, 34, 43, 44, 52],
    'filtered_out_neurons': [10, 21, 22, 29, 44]
}

results_n_bubbles_400 = {
    'iou_vs_epoch': np.array([
        [1, 0.3003, 0.0369, 0.1928, 0.1384, 0.001],
        [2, 0.1836, 0.1806, 0.2042, 0.1673, 0.001],
        [3, 0.1734, 0.2306, 0.1667, 0.2484, 0.001],
        [4, 0.1646, 0.2598, 0.1635, 0.2739, 0.001],
        [5, 0.1615, 0.2743, 0.1607, 0.2842, 0.001],
        [6, 0.1588, 0.2842, 0.1584, 0.2986, 0.001],
        [7, 0.1571, 0.2903, 0.1554, 0.3080, 0.001],
        [8, 0.1560, 0.2942, 0.1542, 0.3045, 0.001],
        [9, 0.1552, 0.2971, 0.1548, 0.3090, 0.001],
        [10, 0.1569, 0.2985, 0.1544, 0.3036, 0.001],
        [11, 0.1547, 0.3014, 0.1531, 0.3094, 0.001],
        [12, 0.1535, 0.3041, 0.1535, 0.3117, 0.001],
        [13, 0.1563, 0.3049, 0.1527, 0.3097, 0.001],
        [14, 0.1528, 0.3081, 0.1520, 0.3213, 0.001],
        [15, 0.1522, 0.3105, 0.1512, 0.3086, 0.001],
        [16, 0.1519, 0.3121, 0.1501, 0.3207, 0.001],
        [17, 0.1515, 0.3137, 0.1511, 0.3140, 0.001],
        [18, 0.1512, 0.3154, 0.1513, 0.3196, 0.001],
        [19, 0.1510, 0.3165, 0.1505, 0.3136, 0.001],
        [20, 0.1507, 0.3180, 0.1517, 0.3042, 0.001],
        [21, 0.1506, 0.3193, 0.1505, 0.3014, 0.001],
        [22, 0.1512, 0.3198, 0.1510, 0.3281, 0.001],
        [23, 0.1500, 0.3222, 0.1499, 0.3191, 0.001],
        [24, 0.1513, 0.3223, 0.1503, 0.3146, 0.001],
        [25, 0.1496, 0.3242, 0.1485, 0.3238, 0.001],
        [26, 0.1491, 0.3258, 0.1486, 0.3171, 0.001],
        [27, 0.1491, 0.3266, 0.1491, 0.3162, 0.001],
        [28, 0.1489, 0.3274, 0.1485, 0.3167, 0.001],
        [29, 0.1489, 0.3278, 0.1486, 0.3313, 0.001],
        [30, 0.1484, 0.3296, 0.1482, 0.3238, 0.001],
        [31, 0.1486, 0.3296, 0.1488, 0.3159, 0.001],
        [32, 0.1375, 0.3447, 0.1368, 0.3489, 0.0001],
        [33, 0.1370, 0.3463, 0.1365, 0.3517, 0.0001],
        [34, 0.1368, 0.3471, 0.1364, 0.3537, 0.0001],
        [35, 0.1367, 0.3478, 0.1363, 0.3549, 0.0001],
        [36, 0.1365, 0.3484, 0.1362, 0.3565, 0.0001],
        [37, 0.1364, 0.3490, 0.1360, 0.3579, 0.0001],
        [38, 0.1362, 0.3495, 0.1360, 0.3581, 0.0001],
        [39, 0.1361, 0.3500, 0.1359, 0.3597, 0.0001],
        [40, 0.1360, 0.3505, 0.1359, 0.3608, 0.0001],
        [41, 0.1359, 0.3510, 0.1358, 0.3615, 0.0001],
        [42, 0.1358, 0.3514, 0.1357, 0.3618, 0.0001],
        [43, 0.1357, 0.3518, 0.1357, 0.3621, 0.0001],
        [44, 0.1356, 0.3523, 0.1356, 0.3630, 0.0001],
        [45, 0.1355, 0.3526, 0.1355, 0.3632, 0.0001],
        [46, 0.1354, 0.3530, 0.1354, 0.3635, 0.0001],
        [47, 0.1353, 0.3534, 0.1353, 0.3633, 0.0001],
        [48, 0.1352, 0.3537, 0.1352, 0.3632, 0.0001],
        [49, 0.1352, 0.3541, 0.1351, 0.3629, 0.0001],
        [50, 0.1351, 0.3544, 0.1351, 0.3631, 0.0001],
    ]),
    'gain_vs_c_len': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0289, 1.0363, 1.0381, 1.0320]),
        'std_gain': np.array([0.3977, 0.4832, 0.5210, 0.5128, 0.5126])
    },
    'gain_vs_c_len_11x11': {
        'c_len': np.array([1, 3, 5, 7, 9]),
        'mean_gain': np.array([1.0000, 1.0143, 1.0195, 1.0158, 1.0191]),
        'std_gain': np.array([0.1030, 0.0718, 0.0737, 0.0719, 0.0720])
    },
    'gain_vs_spacing': {
        'spacing': np.array([1.00, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86, 2.00]),
        'mean_gain': np.array([1.0283, 1.0498, 1.0660, 1.0617, 1.0747, 1.0851, 1.1135, 1.1250]),
        'std_gain': np.array([0.2252, 0.1600, 0.1656, 0.1230, 0.1294, 0.1132, 0.1256, 0.1167])
    },
    'no_optimal_stimuli_neurons': [2, 10, 13, 19, 32, 34, 37, 42, 43, 56, 57],
    'filtered_out_neurons': [4, 35, 48]
}

if __name__ == "__main__":
    plt.ion()

    _, axis = plt.subplots()

    plotting_fcns.plot_iou_results(results_n_bubbles_0, ax=axis, label='0_bubbles', c='b')
    plotting_fcns.plot_iou_results(results_n_bubbles_50, ax=axis, label='50_bubbles', c='r')
    plotting_fcns.plot_iou_results(results_n_bubbles_100, ax=axis, label='100_bubbles', c='g')
    plotting_fcns.plot_iou_results(results_n_bubbles_200, ax=axis, label='200_bubbles', c='m')
    plotting_fcns.plot_iou_results(results_n_bubbles_300, ax=axis, label='300_bubbles', c='c')
    plotting_fcns.plot_iou_results(results_n_bubbles_400, ax=axis, label='400_bubbles', c='k')

    plt.legend()
    plt.title("IoU vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid()

    # Contour Length
    _, axis = plt.subplots()

    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_0, ax=axis, label='0_bubbles', c='b')
    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_50, ax=axis, label='50_bubbles', c='r')
    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_100, ax=axis, label='100_bubbles', c='g')
    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_200, ax=axis, label='200_bubbles', c='m')
    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_300, ax=axis, label='300_bubbles', c='c')
    plotting_fcns.plot_gain_vs_contour_len(
        results_n_bubbles_400, ax=axis, label='400_bubbles', c='k')

    plt.legend()
    plt.title("Contour Gain vs Length")
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # Contour Length 11x11
    _, axis = plt.subplots()

    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_0, ax=axis, label='0_bubbles', c='b')
    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_50, ax=axis, label='50_bubbles', c='r')
    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_100, ax=axis, label='100_bubbles', c='g')
    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_200, ax=axis, label='200_bubbles', c='m')
    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_300, ax=axis, label='300_bubbles', c='c')
    plotting_fcns.plot_gain_vs_contour_len_11x11(
        results_n_bubbles_400, ax=axis, label='400_bubbles', c='k')

    plt.legend()
    plt.title("Contour Gain vs Length - Frag size 11x11")
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # Fragment Spacing
    _, axis = plt.subplots()

    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_0, ax=axis, label='0_bubbles', c='b')
    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_50, ax=axis, label='50_bubbles', c='r')
    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_100, ax=axis, label='100_bubbles', c='g')
    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_200, ax=axis, label='200_bubbles', c='m')
    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_300, ax=axis, label='300_bubbles', c='c')
    plotting_fcns.plot_gain_vs_fragment_spacing(
        results_n_bubbles_400, ax=axis, label='400_bubbles', c='k')

    plt.legend()
    plt.title("Contour Gain vs Fragment Spacing")
    plt.xlabel("Fragment Spacing (RCD)")
    plt.ylabel("Gain")
    plt.grid()
    axis.set_ylim(bottom=0)

    # -----------------------------------------------------------------------------------
    import pdb

    pdb.set_trace()
