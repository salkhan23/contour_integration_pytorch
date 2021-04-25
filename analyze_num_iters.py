# -------------------------------------------------------------------------------------
#  Call contour data set training script on different number of recurrent iterations
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models


if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)

    n_iters_arr = [1, 2, 3, 5, 8, 10, 15, 20, 25]
    # n_iters_arr = n_iters_arr[::-1]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'random_seed': random_seed,
        'train_batch_size': 32,
        'test_batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'lateral_w_reg_weight': 0.0001,
        'lateral_w_reg_gaussian_sigma': 10,
        'clip_negative_lateral_weights': True,
        'lr_sched_step_size': 80,
        'lr_sched_gamma': 0.5
    }

    for n_iters in n_iters_arr:
        print("Processing num_iters = {} {}".format(n_iters, '*'*40))

        base_results_dir = './results/num_iteration_explore_2/n_iters_{}'.format(n_iters)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=n_iters, use_recurrent_batch_norm=True)
        model = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
