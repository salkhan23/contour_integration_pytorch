import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import models.new_piech_models as new_piech_models
import dataset
import utils

contour_integration_in_act = 0
contour_integration_out_act = 0


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer"""
    global contour_integration_in_act
    global contour_integration_out_act

    contour_integration_in_act = layer_in[0].cpu().detach().numpy()
    contour_integration_out_act = layer_out.cpu().detach().numpy()


def get_tgt_neuron_acts(model, tgt_neuron, data_loader):

    global contour_integration_in_act
    global contour_integration_out_act

    model.eval()
    detect_thresh = 0.5
    e_iou = 0

    tgt_neuron_in_acts = []
    tgt_neuron_out_acts = []

    with torch.no_grad():
        for iteration, (img, label) in enumerate(data_loader, 0):
            img = img.to(device)
            label = label.to(device)

            # Zero All activations before each iteration
            contour_integration_in_act = 0
            contour_integration_out_act = 0

            label_out = model(img)
            preds = (label_out > detect_thresh)

            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

            # Store the activations
            _, _, r, c = contour_integration_in_act.shape
            tgt_neuron_in_acts.append(contour_integration_in_act[0, tgt_neuron, r // 2, c // 2])
            tgt_neuron_out_acts.append(contour_integration_out_act[0, tgt_neuron, r // 2, c // 2])

            # # show the image
            # import pdb
            # pdb.set_trace()

    e_iou = e_iou / len(data_loader)

    return e_iou, np.array(tgt_neuron_in_acts), np.array(tgt_neuron_out_acts)


def get_contour_gain_vs_length(
        model, c_len_arr, beta_arr, data_folder, params_idx, full_tile_size, tgt_neuron, epsilon=1e-5):

    iou_scores = []
    cont_int_gain_means = []
    cont_int_gain_stds = []

    # Common
    # ----------------------
    # # Normalization for Dataset
    # mean = meta_data['channel_mean']
    # std = meta_data['channel_std']

    # Normalization used during training
    mean = np.array([0.36167968, 0.3632432, 0.36181496])
    std = np.array([0.43319716, 0.43500246, 0.43306438])

    normalize = transforms.Normalize(mean=mean, std=std)

    for c_idx, c_len in enumerate(c_len_arr):
        # Data Loader
        # ----------
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

        iou_scores.append(avg_iou)

        cont_int_gain_arr = tgt_n_out_arr / (tgt_n_in_arr + epsilon)

        # plt.figure()
        # plt.plot(cont_int_gain_arr)
        #
        # import pdb
        # pdb.set_trace()

        cont_int_gain_means.append(cont_int_gain_arr.mean())
        cont_int_gain_stds.append(cont_int_gain_arr.std())

    return iou_scores, np.array(cont_int_gain_means), np.array(cont_int_gain_stds)


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


def plot_iou_vs_contour_length(c_len_arr, iou_arr, store_dir, fig_name, title=None):
    """
     Plot and Save Gain vs Contour Length

    :param title:
    :param fig_name: What to use for label
    :param c_len_arr:
    :param iou_arr:
    :param store_dir:
    :return:
    """
    f = plt.figure()
    plt.plot(c_len_arr, iou_arr)
    plt.xlabel("Contour Length")
    plt.ylabel("IoU")
    if title is not None:
        plt.title("{}".format(fig_name))
    f.savefig(os.path.join(store_dir, 'iou_{}.jpg'.format(fig_name)), format='jpg')
    plt.close()


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

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    frag_size = np.array([7, 7])
    image_size = np.array([256, 256, 3])

    results_dir = './results/li_2006_experiments'

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Setting up the Model ...")
    net = new_piech_models.ContourIntegrationCSI(lateral_e_size=23, lateral_i_size=23)
    saved_model = './results/new_model/ContourIntegrationCSI_20191005_195738_base/best_accuracy.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # Register Callback
    net.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # Actual folder to store the results
    results_store_dir = os.path.join(
        results_dir,
        os.path.dirname(saved_model).split('/')[-1])
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -----------------------------------------------------------------------------------
    # Data Source
    # -----------------------------------------------------------------------------------
    print("====> Data Initialization")

    # In this data set an optimal stimulus is defined for each channel
    data_set_dir = "./data/channel_wise_optimal_full14_frag7_centered_test"
    print("Source: {}".format(data_set_dir))

    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)

    data_dir = os.path.join(data_set_dir, 'val')
    n_images_per_set = meta_data['n_val_images_per_set']

    bg_tile_size = meta_data["full_tile_size"]

    # -----------------------------------------------------------------------------------
    # Main Routine
    # -----------------------------------------------------------------------------------
    contour_len_arr = [1, 3, 5, 7, 9]
    beta_rotation_arr = [0]

    gain_vs_len_dir = os.path.join(results_store_dir, 'gain_vs_len')
    if not os.path.exists(gain_vs_len_dir):
        os.makedirs(gain_vs_len_dir)

    iou_vs_len_dir = os.path.join(results_store_dir, 'iou_vs_len')
    if not os.path.exists(iou_vs_len_dir):
        os.makedirs(iou_vs_len_dir)

    mean_gain_results = []
    std_gain_results = []
    iou_results = []

    for gp_idx, gabor_params in enumerate(meta_data['g_params_list']):

        print(" {0} Param Set {1} {0}".format('*'*20, gp_idx))

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

        # Channel-wise Results
        # ---------------------
        fig_name = 'neuron_{}'.format(target_neuron)
        img_title = fig_name + ' ff_act {:0.2f} isMaxResponsive {}'.format(
            gabor_params[0]['optimal_stimulus_act'], gabor_params[0]['is_max_active'])

        plot_gain_vs_contour_length(contour_len_arr, mean_gains, std_gains, gain_vs_len_dir, fig_name, img_title)
        plot_iou_vs_contour_length(contour_len_arr, ious, iou_vs_len_dir, fig_name, img_title)

    # -----------------------------------------------------------------------------------
    # Population Results  (n_channels, c_len_arr). Avg over images already done
    # -----------------------------------------------------------------------------------
    mean_gain_results = np.array(mean_gain_results)
    std_gain_results = np.array(std_gain_results)
    iou_results = np.array(iou_results)

    # Identify Outliers
    gain_outlier_threshold = 100
    outliers = [idx for idx, item in enumerate(mean_gain_results) if np.any(item > gain_outlier_threshold)]
    print("{} Outliers (gain > {}) detected. @ {}".format(len(outliers), gain_outlier_threshold, outliers))

    # [1] Overall - Average for all Neurons
    # --------------------------------------
    # # Raw
    # all_iou, all_gain_mu, all_gain_sigma = get_averaged_results(iou_results, mean_gain_results, std_gain_results)
    # fig_name = 'all_neurons_raw'
    # plot_gain_vs_contour_length(contour_len_arr, all_gain_mu, all_gain_sigma, results_store_dir, fig_name)
    # plot_iou_vs_contour_length(contour_len_arr, all_iou, results_store_dir, fig_name)

    # Remove Outliers
    filtered_mu_gains = []
    filtered_sigma_gains = []
    filtered_iou = []

    for n_idx in range(mean_gain_results.shape[0]):
        if n_idx not in outliers:
            filtered_mu_gains.append(mean_gain_results[n_idx, ])
            filtered_sigma_gains.append(std_gain_results[n_idx, ])
            filtered_iou.append(iou_results[n_idx, ])

    all_iou, all_gain_mu, all_gain_sigma = get_averaged_results(filtered_iou, filtered_mu_gains, filtered_sigma_gains)

    fig_name = 'all_neurons'
    plot_gain_vs_contour_length(contour_len_arr, all_gain_mu, all_gain_sigma, results_store_dir, fig_name)
    plot_iou_vs_contour_length(contour_len_arr, all_iou, results_store_dir, fig_name)

    # [2] Only neuron that are max active for their preferred Stimuli
    # --------------------------------------------------------------
    max_active_neurons = [idx for idx, item in enumerate(meta_data['g_params_list']) if item[0]['is_max_active']]

    filtered_mean_gain_results = []
    filtered_std_gain_results = []
    filtered_iou = []

    for n_idx in max_active_neurons:
        if n_idx not in outliers:
            filtered_mean_gain_results.append(mean_gain_results[n_idx, ])
            filtered_std_gain_results.append(std_gain_results[n_idx, ])
            filtered_iou.append(iou_results[n_idx, ])

    max_active_iou, max_active_gain_mu, max_active_gain_sigma = \
        get_averaged_results(filtered_mean_gain_results, filtered_std_gain_results, filtered_iou)

    fig_name = 'Only max_active_neurons'
    title = '{} Neurons are max active for their preferred stimuli'.format(len(max_active_neurons))

    plot_gain_vs_contour_length(
        contour_len_arr, max_active_gain_mu, max_active_gain_sigma, results_store_dir, fig_name, title)
    plot_iou_vs_contour_length(contour_len_arr, max_active_iou, results_store_dir, fig_name)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
