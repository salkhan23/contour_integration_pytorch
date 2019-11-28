# -------------------------------------------------------------------------------------
#  Call contour data set training script on different tau values
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
from models.new_piech_models import ContourIntegrationCSI
from models.piech_models import CurrentSubtractiveInhibition


if __name__ == '__main__':
    random_seed = 5
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    tau_arr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 16,
        'test_batch_size': 1,
        'learning_rate': 3e-5,
        'num_epochs': 50,
    }

    for tau in tau_arr:
        print("Processing tau = {} {}".format(tau, '*'*40))

        base_results_dir = './results/tau_explore/tau_{}'.format(tau)

        model = ContourIntegrationCSI(n_iters=5, lateral_e_size=23, lateral_i_size=15, a=tau, b=tau)
        # model = CurrentSubtractiveInhibition(
        #     edge_out_ch=64, n_iters=3, lateral_e_size=15, lateral_i_size=15, a=tau, b=tau)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()
