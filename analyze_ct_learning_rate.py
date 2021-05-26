# -------------------------------------------------------------------------------------
# Calls contour tracing in natural images script with different number of
# puncturing bubbles
# -------------------------------------------------------------------------------------
import numpy as np
import torch

from train_contour_trace_natural_images import main
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models


if __name__ == '__main__':

    lr_arr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    data_set_parameters = {
        'data_set_dir': './data/pathfinder_natural_images_2',
    }

    for lr in lr_arr:

        train_parameters = {
            'random_seed': 15,
            'train_batch_size': 32,
            'test_batch_size': 32,
            'learning_rate': lr,
            'num_epochs': 100,
            'lateral_w_reg_weight': 0.0001,
            'lateral_w_reg_gaussian_sigma': 10,
            'clip_negative_lateral_weights': True,
            'lr_sched_step_size': 80,
            'lr_sched_gamma': 0.5,
            'punc_n_bubbles': 200,
            'punc_fwhm': np.array([7, 9, 11, 13, 15, 17])
        }

        # # Create Model
        # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
        # cont_int_layer = new_piech_models.CurrentDivisiveInhibitLayer(
        #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)

        cont_int_layer = new_control_models.ControlMatchParametersLayer(
            lateral_e_size=15, lateral_i_size=15)
        # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
        #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
        # cont_int_layer = new_control_models.ControlRecurrentCnnLayer(
        #     lateral_e_size=15, lateral_i_size=15, n_iters=5)

        scale_down_input_to_contour_integration_layer = 4
        net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)

        main(
            net,
            train_params=train_parameters,
            data_set_params=data_set_parameters,
            base_results_store_dir=
                './results/contour_tracing_sensitivity_analysis/learning_rate/lr_{}'.format(lr),
            cont_int_scale=scale_down_input_to_contour_integration_layer,
            n_imgs_for_exp=5000
        )

    # ----------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
