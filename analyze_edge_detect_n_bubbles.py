# ---------------------------------------------------------------------------------------
# Call edge detection on natural image detection with gaussian puncturing
# Bubble Size is varied
# ---------------------------------------------------------------------------------------
import torch

from train_edge_data_set import main
import models.new_piech_models as new_piech_models

if __name__ == '__main__':
    random_seed = 10
    torch.manual_seed(random_seed)

    n_bubbles_arr = [0, 50, 100, 200, 300]

    for n_bubbles in n_bubbles_arr:
        print("{0} Processing num bubble = {1}, {0}".format('*' * 20, n_bubbles))

        data_set_parameters = {
            'data_set_dir': './data/edge_detection_data_set_canny_dynamic',
            'train_subset_size': 30000,
            'test_subset_size': None,
            'n_bubbles': n_bubbles,
            'bubble_fwhm': 11,
        }

        train_parameters = {
            'train_batch_size': 32,
            'test_batch_size': 1,
            'learning_rate': 1e-3,
            'num_epochs': 50,
            'gaussian_reg_weight': 0.0001,
            'gaussian_reg_sigma': 10,
        }

        cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
            lateral_e_size=15, lateral_i_size=15, n_iters=5)

        net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

        main(net, train_params=train_parameters, data_set_params=data_set_parameters,
             base_results_store_dir='./results/explore_n_bubbles')
