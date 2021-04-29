# -------------------------------------------------------------------------------------
#  Call contour data set training script with different tau (a, b) values
# NOTE: a sigmoid is applied on tau before using
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models
from train_utils import inverse_sigmoid


if __name__ == '__main__':
    random_seed = 5
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    sigma_tau_arr = [0.001, 0.2, 0.4, 0.6, 0.8, 0.999]
    a_b_arr = inverse_sigmoid(sigma_tau_arr)

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

    for tau in a_b_arr:
        print("Processing a_b = {} {}".format(tau, '*'*40))

        base_results_dir = './results/tau_explore/a_{:0.3f}'.format(tau)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=5, a=tau, use_recurrent_batch_norm=True)
        model = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

        main(
            model,
            data_set_params=data_set_parameters,
            train_params=train_parameters,
            base_results_store_dir=base_results_dir
        )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
