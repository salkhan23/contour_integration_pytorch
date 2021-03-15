# -------------------------------------------------------------------------------------
#  Call contour data set training script with different learning rates
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models


if __name__ == '__main__':
    random_seed = 10

    lr_arr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    for lr in lr_arr:
        print("Processing learning rate = {} {}".format(lr, '*'*40))

        base_results_dir = './results/analyze_lr_rate/lr_{}'.format(lr)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
        model = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

        train_parameters = {
            'random_seed': random_seed,
            'train_batch_size': 32,
            'test_batch_size': 32,
            'learning_rate': lr,
            'num_epochs': 100,
            'lateral_w_reg_weight': 0.0001,
            'lateral_w_reg_gaussian_sigma': 10,
            'clip_negative_lateral_weights': True,
            'lr_sched_step_size': 100,
            'lr_sched_gamma': 0.5
        }

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
