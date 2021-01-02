# ---------------------------------------------------------------------------------------
# Call train contour Data set training script with different Fix Initializations of
# J_xy and J_yx
# ---------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_data_set import main
import models.new_piech_models as new_piech_models
from train_utils import inverse_sigmoid

if __name__ == '__main__':
    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    sigma_jxy_jyx_range = np.array([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999])
    jxy_jyx_range = inverse_sigmoid(sigma_jxy_jyx_range)

    # ----------------------------------------------------------------------
    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 32,
        'test_batch_size': 1,
        'learning_rate': 1e-4,
        'lr_sched_step_size': 40,
        'num_epochs': 60,
        'gaussian_reg_weight': 0.0001,
        'gaussian_reg_sigma': 10,
        'clip_negative_lateral_weights': False
    }

    for jxy_jyx in jxy_jyx_range:
        print("Processing jxy_jyx = {}".format(jxy_jyx, '*' * 40))

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=5, j_xy=jxy_jyx, j_yx=jxy_jyx)

        net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

        main(net, train_params=train_parameters, data_set_params=data_set_parameters,
             base_results_store_dir='./results/explore_fixed_jxy_jyx/jxy_jyx_{:0.3}'.format(jxy_jyx))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
