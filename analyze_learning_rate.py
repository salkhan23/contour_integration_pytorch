# -------------------------------------------------------------------------------------
#  Call contour data set training script on different number of iterations
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
from models.new_piech_models import ContourIntegrationAlexnet


if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    lr_arr = [0.00001, 0.00002, 0.00003, 0.000005, 0.000001]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    for lr in lr_arr:
        print("Processing learning rate = {} {}".format(lr, '*'*40))

        base_results_dir = './results/analyze_lr_rate/lr_{}'.format(lr)

        model = ContourIntegrationAlexnet(n_iters=5, lateral_e_size=15, lateral_i_size=15)

        train_parameters = {
            'train_batch_size': 16,
            'test_batch_size': 1,
            'learning_rate': lr,
            'num_epochs': 50,
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
