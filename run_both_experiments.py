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

    # # Contour Dataset
    # # ----------------
    # contour_integration_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # net = new_piech_models.ContourIntegrationResnet50(contour_integration_layer)
    # saved_model = \
    #     './results/new_model_resnet_based/Old' \
    #     '/ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20200816_222302_baseline' \
    #     '/best_accuracy.pth'
    # replacement_layer = None
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.load_state_dict(torch.load(saved_model, map_location=device))
    # get_iou_results = True

    # Contour Tracing Natural Images
    # ------------------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    saved_model = \
        './results/contour_tracing' \
        '/BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20210308_090110_pos_only_lr_1e_3_new_classifier_head_recurrent_BN_as_defined_in_paper' \
        '/best_accuracy.pth'
    replacement_layer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))
    get_iou_results = False

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
    #     new_piech_models.CurrentSubtractInhibitLayer(
    #         lateral_e_size=15, lateral_i_size=15, n_iters=5)
    #
    # net = new_piech_models.EdgeDetectionResnet50(contour_integration_layer)
    # saved_model = \
    #     './results/edge_detection_new' \
    #     '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_punctured50_20200312_161536' \
    #     '/best_accuracy.pth'
    # net.load_state_dict(torch.load(saved_model))
    # replacement_layer = None
    # get_iou_results = False

    # # Model jointly trained on contour and pathfinder data sets
    # # ---------------------------------------------------------
    # contour_integration_layer = \
    #     new_piech_models.CurrentSubtractInhibitLayer(
    #         lateral_e_size=15, lateral_i_size=15, n_iters=5)
    #
    # net = new_piech_models.JointPathfinderContourResnet50(
    #     contour_integration_layer)
    # saved_model = \
    #     'results/joint_training/' \
    #     'JointPathfinderContourResnet50_CurrentSubtractInhibitLayer_20200713_230237_first_run/' \
    #     'last_epoch.pth'
    #
    # replacement_layer = None
    # dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.load_state_dict(torch.load(saved_model, map_location=dev))
    # get_iou_results = True

    # ------------------------------------
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    results_dir = os.path.dirname(saved_model)

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    frag_size_list = [(7, 7)]
    # frag_size_list = [(7, 7), (9, 9), (11, 11), (13, 13)]

    for frag_size in frag_size_list:
        print("Processing Fragment Size {} {}".format(frag_size, '-'*50))
        frag_size = np.array(frag_size)

        print("Getting contour length results")
        optim_stim_dict = experiment_gain_vs_len.main(
            net,
            results_dir,
            iou_results=get_iou_results,
            embedded_layer_identifier=replacement_layer,
            frag_size=frag_size)

        print("Getting fragment spacing results")
        experiment_gain_vs_spacing.main(
            net, results_dir,
            embedded_layer_identifier=replacement_layer,
            frag_size=frag_size,
            optimal_stim_dict=optim_stim_dict)

    # -----------------------------------------------------------------------------------
    print("Running script took {}".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
