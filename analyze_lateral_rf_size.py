# -------------------------------------------------------------------------------------
#  Call contour data set training script with different Lateral weight sizes.
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models


if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    lateral_rf_size_arr = [3, 5, 7, 11, 15, 19, 23, 27, 31, 35]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 32,
        'test_batch_size': 1,
        'learning_rate': 3e-5,
        'num_epochs': 50,
        'lateral_w_reg_weight': 0.0001,
        'lateral_w_reg_gaussian_sigma': 10,
    }

    for lateral_rf_size in lateral_rf_size_arr:
        print("Processing lateral rf_size = {} {}".format(lateral_rf_size, '*'*40))

        base_results_dir = './results/lateral_rf_explore/size_{}'.format(lateral_rf_size)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=lateral_rf_size, lateral_i_size=lateral_rf_size, n_iters=5)
        model = new_piech_models.ContourIntegrationAlexnet(cont_int_layer)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
