# -------------------------------------------------------------------------------------
#  Call contour data set training script with different Lateral weight sizes.
# -------------------------------------------------------------------------------------
from train_contour_data_set import main
import models.new_piech_models as new_piech_models


if __name__ == '__main__':

    lateral_rf_size_arr = [3, 5, 7, 11, 15, 19, 23, 27, 31, 35]

    data_set_parameters = {
        'data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'random_seed': 1,
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

    for lateral_rf_size in lateral_rf_size_arr:
        print("Processing lateral rf_size = {} {}".format(lateral_rf_size, '*'*40))

        base_results_dir = './results/lateral_rf_explore/size_{}'.format(lateral_rf_size)

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=lateral_rf_size,
            lateral_i_size=lateral_rf_size,
            n_iters=5,
            use_recurrent_batch_norm=True
        )

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
