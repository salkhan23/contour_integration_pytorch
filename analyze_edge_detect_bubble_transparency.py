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

    transparency_arr_arr = [0, 0.25, 0.5, 0.75, 1]

    for bubble_transparency in transparency_arr_arr:
        print("{0} Processing bubble transparency {1}, {0}".format('*'*20, bubble_transparency))

        data_set_parameters = {
            'data_set_dir': './data/edge_detection_data_set_canny_dynamic',
            'train_subset_size': 30000,
            'test_subset_size': None,
            'n_bubbles': 100,
            'bubble_fwhm': 20,
            'bubble_transparency': bubble_transparency
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
             base_results_store_dir='./results/explore_bubble_transparency')
