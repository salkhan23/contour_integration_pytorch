# -------------------------------------------------------------------------------------
# Call contour data set training script (v3) with different lateral kernel gaussian
# regularization sigmas (widths)
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set_3 import main
from models.new_piech_models import ContourIntegrationCSI

if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    gaussian_reg_weight_arr = [0.01, 0.0001, 0.00001, 0.000001]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    for loss_weight in gaussian_reg_weight_arr:
        print("Processing gaussian regularization with width = {} {}".format(loss_weight, '*' * 40))

        train_parameters = {
            'train_batch_size': 16,
            'test_batch_size': 1,
            'learning_rate': 0.00003,
            'num_epochs': 50,
            'gaussian_reg_weight': loss_weight,
            'gaussian_reg_sigma': 6
        }

        base_results_dir = './results/gaussian_reg_loss_weight_explore/weight_{}'.format(loss_weight)

        model = ContourIntegrationCSI(n_iters=5, lateral_e_size=15, lateral_i_size=15)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
