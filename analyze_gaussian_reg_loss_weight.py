# -------------------------------------------------------------------------------------
# Calls contour data set training script with different gaussian regularization
# loss weights.
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models

if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    gaussian_reg_weight_arr = [0.1, 0.01, 0.0001, 0.00001, 0.000001]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    for loss_weight in gaussian_reg_weight_arr:
        print("Processing gaussian regularization with width = {} {}".format(loss_weight, '*' * 40))

        train_parameters = {
            'random_seed': random_seed,
            'train_batch_size': 32,
            'test_batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'lateral_w_reg_weight': loss_weight,
            'lateral_w_reg_gaussian_sigma': 10,
            'clip_negative_lateral_weights': True,
            'lr_sched_step_size': 80,
            'lr_sched_gamma': 0.5
        }

        base_results_dir = './results/gaussian_reg_loss_weight_explore/weight_{}'.format(loss_weight)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
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
