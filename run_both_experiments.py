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

    # Model
    # -------
    # # Model trained with 5 iterations
    # net = new_piech_models.ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # saved_model = './results/new_model/ContourIntegrationCSI_20200130_181122_gaussian_reg_sigma_10_loss_e-5/' \
    #               'best_accuracy.pth'

    net = new_piech_models.get_embedded_resnet50_model()
    saved_model = './results/imagenet_classification/ResNet_20200208_153412/' \
                  'best_accuracy.pth'
    checkpoint = torch.load(saved_model)
    net.load_state_dict(checkpoint['state_dict'])

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    # Loading Saved Model
    print("===> Loading Model ...")
    # net.load_state_dict(torch.load(saved_model))
    results_dir = os.path.dirname(saved_model)

    experiment_gain_vs_len.main(net, results_dir, iou_results=False, embedded_layer_identifier=net.conv1)
    experiment_gain_vs_spacing.main(net, results_dir, embedded_layer_identifier=net.conv1)

    # -----------------------------------------------------------------------------------
    print("Running script took {}".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
