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

from models.new_piech_models import ContourIntegrationCSI

if __name__ == "__main__":
    random_seed = 10

    # Model
    # -------
    # Model trained with 5 iterations
    net = ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    saved_model = './results/new_model/ContourIntegrationCSI_20200130_181122_gaussian_reg_sigma_10_loss_e-5/' \
                  'best_accuracy.pth'

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    net.load_state_dict(torch.load(saved_model))
    results_dir = os.path.dirname(saved_model)

    experiment_gain_vs_len.main(net, results_dir)
    experiment_gain_vs_spacing.main(net, results_dir)

    # -----------------------------------------------------------------------------------
    print("Running script took {}".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
