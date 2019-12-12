# ----------------------------------------------------------------------
#  Li-2006 Experiment Contour Length vs Gain for Straight Contours
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import models.new_piech_models as new_piech_models
import dataset
import utils

cont_int_in_act = 0
cont_int_out_act = 0


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer
        Callback to Retrieve the activations input & output of the contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


def get_tgt_neuron_acts(model, tgt_neuron, data_loader):

    global cont_int_in_act
    global cont_int_out_act

    model.eval()
    detect_thresh = 0.5
    e_iou = 0

    tgt_neuron_in_acts = []
    tgt_neuron_out_acts = []

    with torch.no_grad():
        for iteration, (img, label) in enumerate(data_loader, 0):
            img = img.to(device)
            label = label.to(device)

            # Zero all activations before each iteration
            cont_int_in_act = 0
            cont_int_out_act = 0

            label_out = model(img)
            preds = (label_out > detect_thresh)

            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

            # Store the activations
            _, _, r, c = cont_int_in_act.shape
            tgt_neuron_in_acts.append(cont_int_in_act[0, tgt_neuron, r // 2, c // 2])
            tgt_neuron_out_acts.append(cont_int_out_act[0, tgt_neuron, r // 2, c // 2])

            # # show the image
            # import pdb
            # pdb.set_trace()

    e_iou = e_iou / len(data_loader)

    return e_iou, np.array(tgt_neuron_in_acts), np.array(tgt_neuron_out_acts)


def plot_gain_vs_contour_length(c_len_arr, mu_gain_arr, sigma_gain_arr, store_dir, f_name, f_title=None):
    """
     Plot and Save Gain vs Contour Length

    :param f_title:
    :param f_name:
    :param c_len_arr:
    :param mu_gain_arr:
    :param sigma_gain_arr:
    :param store_dir:
    :return:
    """
    f = plt.figure()
    plt.errorbar(c_len_arr, mu_gain_arr, sigma_gain_arr)
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    if f_title is not None:
        plt.title("{}".format(f_title))
    f.savefig(os.path.join(store_dir, 'gain_vs_len_{}.jpg'.format(f_name)), format='jpg')
    plt.close()


def plot_iou_vs_contour_length(c_len_arr, iou_arr, store_dir, f_name, f_title=None):
    """
     Plot and Save Gain vs Contour Length

    :param f_title:
    :param f_name: What to use for label
    :param c_len_arr:
    :param iou_arr:
    :param store_dir:
    :return:
    """
    f = plt.figure()
    plt.plot(c_len_arr, iou_arr)
    plt.xlabel("Contour Length")
    plt.ylabel("IoU")
    if f_title is not None:
        plt.title("{}".format(f_name))
    f.savefig(os.path.join(store_dir, 'iou_{}.jpg'.format(f_name)), format='jpg')
    plt.close()


def get_contour_gain_vs_length(
        model, c_len_arr, beta_arr, data_folder, params_idx, full_tile_size, tgt_neuron, epsilon=1e-5):
    """
       For each entry in c_len_arr, retrieve the output activations of the contour integration layer
       Calculcate gain =

    :param model:
    :param c_len_arr:
    :param beta_arr:
    :param data_folder:
    :param params_idx:
    :param full_tile_size:
    :param tgt_neuron: Channel index of centrally located neuron
    :param epsilon:
    :return:
    """

    iou_per_len_arr = []
    tgt_n_out_mat = []  # Stores output of target neuron per input per len

    # Normalization for Dataset (0 mean and 1 std)
    # --------------------------------------
    # mean = meta_data['channel_mean']
    # std = meta_data['channel_std']

    # Normalization used during training
    mean = np.array([0.36167968, 0.3632432, 0.36181496])
    std = np.array([0.43319716, 0.43500246, 0.43306438])

    normalize = transforms.Normalize(mean=mean, std=std)

    for c_idx, c_len in enumerate(c_len_arr):
        print("Setting up data loader for contour length = {}".format(c_len))
        data_set = dataset.Fields1993(
            data_dir=data_folder,
            bg_tile_size=full_tile_size,
            transform=normalize,
            c_len_arr=[c_len],
            beta_arr=beta_arr,
            gabor_set_arr=[params_idx]
        )

        data_loader = DataLoader(data_set, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

        avg_iou, tgt_n_in_arr, tgt_n_out_arr = get_tgt_neuron_acts(model, tgt_neuron, data_loader)

        tgt_n_out_mat.append(tgt_n_out_arr)
        iou_per_len_arr.append(avg_iou)

    # -------------------
    # Gain Calculation
    # -------------------
    tgt_n_out_mat = np.array([tgt_n_out_mat])
    tgt_n_out_mat = np.squeeze(tgt_n_out_mat)

    # In Li2006, Gain was defined as output of neuron / mean output to noise pattern
    # where the noise pattern was defined as optimal stimulus at center of RF and all
    # others fragments were random. This corresponds to resp c_len=x/ mean resp clen=1
    avg_resp_noise_pattern = np.mean(tgt_n_out_mat[0, ])

    # Gain
    tgt_n_out_mat = tgt_n_out_mat / (avg_resp_noise_pattern + epsilon)

    mean_gain_per_len_arr = np.mean(tgt_n_out_mat, axis=1)
    std_gain_per_len_arr = np.std(tgt_n_out_mat, axis=1)

    return iou_per_len_arr, np.array(mean_gain_per_len_arr), np.array(std_gain_per_len_arr)


def get_averaged_results(iou_arr, gain_mu_arr, gain_std_arr, axis=0):
    """ Average Results over multiple runs indexed in the first dimension """
    iou_arr = np.array(iou_arr)
    gain_mu_arr = np.array(gain_mu_arr)
    gain_std_arr = np.array(gain_std_arr)

    iou = np.mean(iou_arr, axis=axis)
    mean_gain = np.mean(gain_mu_arr, axis=axis)

    # Two RVs, X and Y
    # Given mu_x, mu_y, sigma_x, sigma_y
    # sigma (standard deviation) of X + Y = np.sqrt(sigma_x**2 + sigma_y**2)
    std_gain = np.sqrt(np.sum(gain_std_arr ** 2, axis=axis))

    return iou, mean_gain, std_gain


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    results_dir = './results/li_2006_experiment'

    # -------------------------------
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Setting up the Model ...")
    net = new_piech_models.ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15)

    # Load Saved Model
    saved_model = './results/num_iteration_explore_fix_and_sigmoid_gate/' \
                  'n_iters_15/ContourIntegrationCSI_20191210_052543/best_accuracy.pth'

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # Register Callback
    net.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # -----------------------------------------------------------------------------------
    # Actual folder to store the results
    # -----------------------------------------------------------------------------------
    results_store_dir = os.path.join(
        results_dir,
        os.path.dirname(saved_model).split('/')[-1])
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # Sub results folders for individual neuron tuning curves
    gain_vs_len_dir = os.path.join(results_store_dir, 'gain_vs_len')
    if not os.path.exists(gain_vs_len_dir):
        os.makedirs(gain_vs_len_dir)

    iou_vs_len_dir = os.path.join(results_store_dir, 'iou_vs_len')
    if not os.path.exists(iou_vs_len_dir):
        os.makedirs(iou_vs_len_dir)

    # -----------------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------------
    print("====> Data Initialization")

    # In this data set an optimal stimulus is defined for each channel
    data_set_dir = "./data/channel_wise_optimal_full14_frag7_centered"
    print("Source: {}".format(data_set_dir))

    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)

    data_dir = os.path.join(data_set_dir, 'val')

    bg_tile_size = meta_data["full_tile_size"]

    # -----------------------------------------------------------------------------------
    # Main Routine
    # -----------------------------------------------------------------------------------
    contour_len_arr = [1, 3, 5, 7, 9]
    beta_rotation_arr = [0]

    mean_gain_results = []
    std_gain_results = []
    iou_results = []

    for gp_idx, gabor_params in enumerate(meta_data['g_params_list']):
        print(" {0} Param Set {1} {0}".format('*' * 20, gp_idx))

        target_neuron = gp_idx  # one gabor param set for each neuron

        ious, mean_gains, std_gains = get_contour_gain_vs_length(
            net,
            contour_len_arr,
            beta_rotation_arr,
            data_dir,
            gp_idx,
            full_tile_size=bg_tile_size,
            tgt_neuron=target_neuron
        )

        mean_gain_results.append(mean_gains)
        std_gain_results.append(std_gains)
        iou_results.append(ious)

        # Individual Neuron Curves
        fig_name = 'neuron_{}'.format(target_neuron)
        img_title = fig_name + ' ff_act {:0.2f} isMaxResponsive {}'.format(
            gabor_params[0]['optimal_stimulus_act'], gabor_params[0]['is_max_active'])

        plot_gain_vs_contour_length(contour_len_arr, mean_gains, std_gains, gain_vs_len_dir, fig_name, img_title)
        plot_iou_vs_contour_length(contour_len_arr, ious, iou_vs_len_dir, fig_name, img_title)

        # import pdb
        # pdb.set_trace()

    # -----------------------------------------------------------------------------------
    # Population Results (n_channels, c_len_arr).
    # Avg over images already done
    # -----------------------------------------------------------------------------------
    mean_gain_results = np.array(mean_gain_results)
    std_gain_results = np.array(std_gain_results)
    iou_results = np.array(iou_results)

    # Identify Outliers
    gain_outlier_threshold = 50
    outliers = [idx for idx, item in enumerate(mean_gain_results) if np.any(item > gain_outlier_threshold)]
    print("{} Outliers (gain > {}) detected. @ {}".format(len(outliers), gain_outlier_threshold, outliers))

    # [1] Overall - Average All Neurons (without outliers)
    # -------------------------------------------------
    all_neurons = np.arange(mean_gain_results.shape[0])
    filt_all_neurons = [idx for idx in all_neurons if idx not in outliers]

    filt_mu_gains = mean_gain_results[filt_all_neurons, ]
    filt_sigma_gains = std_gain_results[filt_all_neurons, ]
    filt_iou = iou_results[filt_all_neurons, ]
    all_iou, all_gain_mu, all_gain_sigma = get_averaged_results(filt_iou, filt_mu_gains, filt_sigma_gains)

    fig_name = 'average_all_neurons'
    title = 'Average over all neurons'
    plot_gain_vs_contour_length(contour_len_arr, all_gain_mu, all_gain_sigma, results_store_dir, fig_name, title)
    plot_iou_vs_contour_length(contour_len_arr, all_iou, results_store_dir, fig_name, title)

    # [2] Only neurons that are max active to their preferred stimuli
    # ---------------------------------------------------------------
    max_active_neurons = [idx for idx, item in enumerate(meta_data['g_params_list']) if item[0]['is_max_active']]
    filt_max_active_neurons = [idx for idx in max_active_neurons if idx not in outliers]

    filt_mu_gains = mean_gain_results[filt_max_active_neurons, ]
    filt_sigma_gains = std_gain_results[filt_max_active_neurons, ]
    filt_iou = iou_results[filt_max_active_neurons, ]
    max_active_iou, max_active_gain_mu, max_active_gain_sigma = \
        get_averaged_results(filt_iou, filt_mu_gains, filt_sigma_gains)

    fig_name = 'average_max_active_neurons'
    title = 'Average over all neurons max active for their stimuli (N={})'.format(len(max_active_neurons))
    plot_gain_vs_contour_length(
        contour_len_arr, max_active_gain_mu, max_active_gain_sigma, results_store_dir, fig_name, title)
    plot_iou_vs_contour_length(contour_len_arr, max_active_iou, results_store_dir, fig_name, title)

    # [3] Plot gain curves individually in the same figure
    # -----------------------------------------------------
    filt_mu_gains = np.array(filt_mu_gains)
    fig = plt.figure()
    for idx, n_idx in enumerate(filt_max_active_neurons):
        plt.plot(contour_len_arr, filt_mu_gains[idx, ], label=''.format(n_idx))
    plt.xlabel("Length")
    plt.ylabel("Gain")
    plt.title("Individual Gain Curves - Max Active Neurons")
    plt.legend()
    fig.savefig(os.path.join(results_store_dir, 'individual_gain_curves.jpg'), format='jpg')
    plt.close()

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
