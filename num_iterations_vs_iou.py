# -------------------------------------------------------------------------------------
#  Call contour data set training script on different number of iterations
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
from models.new_piech_models import ContourIntegrationCSI


if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    n_iters_arr = [1, 2, 3, 5, 8, 10, 15, 20, 25]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 16,
        'test_batch_size': 1,
        'learning_rate': 0.00003,
        'num_epochs': 50,
    }

    for n_iters in n_iters_arr:
        print("Processing num_iters = {} {}".format(n_iters, '*'*40))

        base_results_dir = './results/num_iteration_explore/n_iters_{}'.format(n_iters)

        model = ContourIntegrationCSI(n_iters=n_iters, lateral_e_size=23, lateral_i_size=23)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
