# ---------------------------------------------------------------------------------------
#  Get the orientation differences between ff and lateral kernels for multiple modeuls
# and plot them in the same grapgh
# ---------------------------------------------------------------------------------------

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import torch

import gabor_fits
import models.new_piech_models as new_piech_models
import utils
import ff_and_lateral_kernel_orientation_differences

if __name__ == '__main__':

    plt.ion()

    base_results_dir = '../results/multiple_runs_contour_dataset/positive_lateral_weights_with_BN_best_gain_curves/'
    saved_models_arr = [
        'run_1',
        'run_2',
        'run_3',
        'run_4',
        'run_5'
    ]

    full_ff_kernels = []
    full_lateral_e_kernels = []
    full_lateral_i_kernels = []

    for run in saved_models_arr:
        results_dir = os.path.join(base_results_dir, run, 'lateral_kernels/preferred_orientations')

        ff_ori_pickle_filename = 'ff_kernels_orientations.pickle'
        ff_ori_pickle_file = os.path.join(results_dir, ff_ori_pickle_filename)

        with open(ff_ori_pickle_file, 'rb') as handle:
            ff_orientations = pickle.load(handle)

        lateral_ori_pickle_filename = 'lateral_kernels_orientation.pickle'
        lateral_ori_pickle_file = os.path.join(results_dir, lateral_ori_pickle_filename)

        with open(lateral_ori_pickle_file, 'rb') as handle:
            lateral_kernels_orientations = pickle.load(handle)

        full_ff_kernels.extend(ff_orientations)
        full_lateral_e_kernels.extend(lateral_kernels_orientations['E'])
        full_lateral_i_kernels.extend(lateral_kernels_orientations['I'])

    ff_and_lateral_kernel_orientation_differences.plot_ff_lat_orientation_differences(
        full_ff_kernels,
        full_lateral_e_kernels,
        full_lateral_i_kernels
    )

    import pdb
    pdb.set_trace()



