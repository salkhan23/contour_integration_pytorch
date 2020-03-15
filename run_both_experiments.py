# ---------------------------------------------------------------------------------------
# For a given Model Run both Li-2006 experiments and saved the results
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

import torch

import experiment_gain_vs_len
import experiment_gain_vs_spacing

import models.new_piech_models as new_piech_models

if __name__ == "__main__":
    random_seed = 10

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    # Loading Saved Model
    print("===> Loading Model ...")

    # Contour Dataset
    # ----------------
    contour_integration_layer = \
        new_piech_models.CurrentSubtractInhibitLayer(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.ContourIntegrationResnet50(contour_integration_layer)
    saved_model = \
        './results/new_model_resnet_based/' \
        'ContourIntegrationCSIResnet50_20200131_194615_gaussian_reg_sigma_10_weight_0.0001' \
        '/best_accuracy.pth'
    replacement_layer = None
    net.load_state_dict(torch.load(saved_model))
    get_iou_results = True

    # # Imagenet
    # # ----------------------
    # net = new_piech_models.get_embedded_resnet50_model()
    # saved_model = \
    #     './results/imagenet_classification/' \
    #     'ResNet_20200212_202439_untrained_gaussian_reg_epochs_12_momentum_0.9_lr_0.1' \
    #     '/best_accuracy.pth'
    # checkpoint = torch.load(saved_model)
    # replacement_layer = net.conv1
    # net.load_state_dict(checkpoint['state_dict'])
    # get_iou_results = False

    # # Edge Dataset Trained Model
    # # ----------------------------
    # contour_integration_layer = \
    #     new_piech_models.CurrentSubtractInhibitLayer(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    #
    # net = new_piech_models.EdgeDetectionResnet50(contour_integration_layer)
    # saved_model = \
    #     './results/edge_detection_new' \
    #     '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_punctured50_20200312_161536' \
    #     '/best_accuracy.pth'
    # net.load_state_dict(torch.load(saved_model))
    # replacement_layer = None
    # get_iou_results = False

    # ------------------------------------
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    results_dir = os.path.dirname(saved_model)

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    frag_size_list = [(7, 7), (11, 11)]
    # frag_size_list = [(7, 7), (9, 9), (11, 11), (13, 13)]

    for frag_size in frag_size_list:
        print("Processing Fragment Size {} {}".format(frag_size, '-'*50))
        frag_size = np.array(frag_size)

        experiment_gain_vs_len.main(
            net, results_dir, iou_results=get_iou_results, embedded_layer_identifier=replacement_layer,
            frag_size=frag_size)

        experiment_gain_vs_spacing.main(
            net, results_dir, embedded_layer_identifier=replacement_layer, frag_size=frag_size)

    # -----------------------------------------------------------------------------------
    print("Running script took {}".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
