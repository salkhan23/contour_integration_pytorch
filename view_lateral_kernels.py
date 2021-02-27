# -----------------------------------------------------------------------------------------
# PLot Feed-forward and Lateral Kernels
# -----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

import torch

import models.new_piech_models as new_piech_models
import utils

if __name__ == '__main__':
    plt.ion()

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    # # Contour Data Set Model
    # # ----------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/positive_weights/recurrent_batch_normalization_best/' \
        '/ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20210121_092722_100_Epochs_BN_before_Relu_best' \
        '/best_accuracy.pth'

    # # # Contour Tracing in Natural Images
    # # # -------------------------------
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    # scale_down_input_to_contour_integration_layer = 4
    # net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    # saved_model = \
    #     './results/contour_tracing/' \
    #     'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20210220_205459_positive_only_lr_1e-4_long_run' \
    #     '/best_accuracy.pth'

    # -----------------------------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))

    # -----------------------------------------------------------------------------------
    # View the feed-forward kernels
    # -----------------------------------------------------------------------------------
    utils.view_ff_kernels(
        net.edge_extract.weight.data.numpy(),
        # results_store_dir=os.path.dirname(saved_model)
    )

    # -----------------------------------------------------------------------------------
    # view lateral Kernels
    # -----------------------------------------------------------------------------------
    utils.view_spatial_lateral_kernels(
        net.contour_integration_layer.lateral_e.weight.data.numpy(),
        net.contour_integration_layer.lateral_i.weight.data.numpy(),
        spatial_func=np.mean
        # results_store_dir=os.path.dirname(saved_model),
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
