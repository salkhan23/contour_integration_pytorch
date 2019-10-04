import pickle

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gabor_fits


# # ---------------------------------------------------------------------------------------
# # 5 BW Gabors File
# # ---------------------------------------------------------------------------------------
# pickle_file = "bw_5_gabors_params.pickle"
# gabor_parameters = [
#     [{'x0': 0, 'y0': 0, 'theta_deg':  60, 'amp': 1, 'sigma': 4.0, 'lambda1':  8, 'psi':  30, 'gamma': 1}],
#     [{'x0': 0, 'y0': 0, 'theta_deg': 130, 'amp': 1, 'sigma': 4.0, 'lambda1':  7, 'psi':   0, 'gamma': 1}],
#     [{'x0': 0, 'y0': 0, 'theta_deg': 110, 'amp': 1, 'sigma': 4.0, 'lambda1': 11, 'psi': 180, 'gamma': 1}],
#     [{'x0': 0, 'y0': 0, 'theta_deg': 150, 'amp': 1, 'sigma': 4.0, 'lambda1':  9, 'psi':   0, 'gamma': 1}],
#     [{'x0': 0, 'y0': 0, 'theta_deg': 20,  'amp': 1, 'sigma': 4.0, 'lambda1': 10, 'psi':   0, 'gamma': 1}],
# ]

# -----------------------------------------------------------------------------------
# New 10 gabors file.
# Fragments look like small line segments. Some are colored (minimally)
#
# A Background pixel value is included for each gabor set. These stimuli were
# generated for smaller fragments [7x7] (Full tile size=[14x14]). When generating
# stimuli of these sizes, explicitly set bg value. Generated fragments look
# like small line segments and do not taper off at tile boundaries. Therefore
# the default technique of averaging fragment pixel @ the boundaries does not give bg
# values. If bg = None, use the default way of generating bg pixels.
#
# Max active Edge Extraction Neurons (Alexnet)
# --------------------------------------------
# [1] Max Active 18, Value = 40.44, Bg = 0
# [2] Max Active 57, Value = 46.50, Bg = 0
# [3] Max Active 41, Value = 46.92, Bg = 0
# [4] Max Active 23, Value = 32.27, Bg = 0
# [5] Max Active 33, Value = 36.25, Bg = 255
# [6] Max Active 13, Value = 32.08, Bg = None
# [7] Max Active 11, Value = 36.54, Bg = 0
# [8] Max Active  3, Value = 37.89, Bg = 0
# [9] Max Active 63, Value = 39.73, Bg = 0
# [10] Max Active 6, Value = 34.97, Bg = 0

# -----------------------------------------------------------------------------------
pickle_file = 'fitted_10_gabors_params.pickle'

gabor_parameters_list = [
    # [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma]
    [
        [   0, -1.03,  27,  0.33, 1.00,  8.25,    0,    0]
    ],
    [
        [1.33,     0,  70,  1.10, 1.47,  9.25,    0,    0]
    ],
    [
        [   0,     0, 150,   0.3, 0.86, 11.60, 0.61,    0]
    ],
    [
        [   0,     0,   0,   0.3,  1.5,   7.2, 0.28,    0]
    ],
    [
        [   0,     0, 160, -0.33, 0.80, 20.19,    0,    0]
    ],
    [
        [0.00,  0.00, -74,  0.29, 1.30,  4.17, 0.73, 0.15],
        [0.00,  0.00, -74,  0.53, 1.30,  4.74, 0.81, 0.17],
        [0.00,  0.00, -74,  0.27, 1.30,  4.39, 1.03, 0.17],
    ],
    [
        [   0, -1.03, 135,  0.33,  1.0,  8.25,    0,    0]
    ],
    [
        [0.00,  0.00,  80, -0.33, 0.80, 20.19,    0, 0.00]
    ],
    [
        [   0,     0, 120,   0.3, 0.86,  8.60, -0.61,   0]
    ],
    [
        [   0,     0, -45, -0.46,  0.9,    25,     1,   0]
    ]
]

bg_list = [0, 0, 0, 0, 255, None, 0, 255, 0, 255]


gabor_parameters = []
for gabor_set in gabor_parameters_list:
    params = gabor_fits.convert_gabor_params_list_to_dict(gabor_set)

    for chan_params in params:
        # Should be a dictionary
        chan_params['bg'] = 0

    gabor_parameters.append(params)

# -----------------------------------------------------------------------------------
# Save the pickle file
with open(pickle_file,  'wb') as handle:
    pickle.dump(gabor_parameters, handle)

with open(pickle_file, 'rb') as handle:
    a = pickle.load(handle)

input("Press any key to Exit")
