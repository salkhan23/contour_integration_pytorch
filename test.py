# ---------------------------------------------------------------------------------------
# Li -2006 Experiment 1: Contour Length vs Contour Integration Gain
# (for straight contours)
# ---------------------------------------------------------------------------------------
import numpy as np
import os
import pickle

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import dataset
import utils

# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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


def get_tgt_neuron_acts(model, n_idx, data_loader):
    """ Retreive activations of the cetner neurons at index n_idx"""

    global cont_int_in_act
    global cont_int_out_act

    model.eval()
    detect_thresh = 0.5
    e_iou = 0

    in_acts = []
    out_acts = []

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
            in_acts.append(cont_int_in_act[0, n_idx, r // 2, c // 2])
            out_acts.append(cont_int_out_act[0, n_idx, r // 2, c // 2])

            # # Debug - Plot input and output feature maps for the target neuron (channel)
            # f, ax_arr = plt.subplots(1, 2)
            # ax_arr[0].imshow(cont_int_in_act[0, n_idx, ])
            # ax_arr[0].set_title("In Act")
            # ax_arr[1].imshow(cont_int_out_act[0, n_idx, ])
            # ax_arr[1].set_title("out Act")
            # f.suptitle("Feature maps @ channel {} for image @ index {}".format(n_idx, iteration))
            # import pdb
            # pdb.set_trace()

            # # debug - Which neuron is most active
            # plt.figure("all center channels")
            #
            # plt.plot(cont_int_in_act[0, :, r // 2, c // 2], label='{}'.format(iteration))
            # plt.legend()
            # import pdb
            # pdb.set_trace()

            max_active_in = np.argmax(cont_int_in_act[0, :, r // 2, c // 2])

    e_iou = e_iou / len(data_loader)

    return e_iou, np.array(in_acts), np.array(out_acts), max_active_in


def get_contour_gain_vs_length(
        model, c_len_arr, beta_arr, data_folder, params_idx, full_tile_size, n_idx, epsilon=1e-5):
    """
    TODO: Complete me
    Contour Integration Gain = response to contour pattern / mean response to noise pattern

    Noise pattern defined as optimal stimulus in classical RF, all other fragment randomly oriented.


    :param model:
    :param c_len_arr:
    :param beta_arr:
    :param data_folder:
    :param params_idx:
    :param full_tile_size:
    :param n_idx: Channel index of centrally located neuron
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
    mean = np.array([0.46247174, 0.46424958, 0.46231144])
    std = np.array([0.46629872, 0.46702369, 0.46620434])

    normalize = transforms.Normalize(mean=mean, std=std)
    max_act_in_n = 0

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

        avg_iou, tgt_n_in_arr, tgt_n_out_arr, max_act_in_n = get_tgt_neuron_acts(model, n_idx, data_loader)

        tgt_n_out_mat.append(tgt_n_out_arr)
        iou_per_len_arr.append(avg_iou)

        # print("Mean in act {}".format(np.mean(tgt_n_in_arr)))
        # print("in act")
        # print(tgt_n_in_arr)
        #
        # print("Mean out act {}".format(np.mean(tgt_n_out_arr)))
        # print("out act")
        # print(tgt_n_out_arr)

    # -------------------
    # Gain Calculation
    # -------------------
    tgt_n_out_mat = np.array(tgt_n_out_mat)

    # In Li2006, Gain was defined as output of neuron / mean output to noise pattern
    # where the noise pattern was defined as optimal stimulus at center of RF and all
    # others fragments were random. This corresponds to resp c_len=x/ mean resp clen=1
    avg_resp_noise_pattern = np.mean(tgt_n_out_mat[0, ])

    # Gain
    tgt_n_out_mat = tgt_n_out_mat / (avg_resp_noise_pattern + epsilon)

    mean_gain_per_len_arr = np.mean(tgt_n_out_mat, axis=1)
    std_gain_per_len_arr = np.std(tgt_n_out_mat, axis=1)

    return iou_per_len_arr, np.array(mean_gain_per_len_arr), np.array(std_gain_per_len_arr), max_act_in_n


def plot_iou_vs_contour_length(c_len_arr, ious, store_dir, f_name, f_title=None):
    """
     Plot and Save Gain vs Contour Length

    :param f_title:
    :param f_name: What to use for label
    :param c_len_arr:
    :param ious:
    :param store_dir:
    :return:
    """
    f = plt.figure()
    plt.plot(c_len_arr, ious)
    plt.xlabel("Contour Length")
    plt.ylabel("IoU")
    plt.ylim(bottom=0, top=1.0)
    if f_title is not None:
        plt.title("{}".format(f_name))
    f.savefig(os.path.join(store_dir, 'iou_{}.jpg'.format(f_name)), format='jpg')
    plt.close()


def get_averaged_results(ious, gain_mu_arr, gain_std_arr, axis=0):
    """
         Average Results over multiple runs indexed in the first dimension
         Each entry itself is averaged value. We want to get the average mu and sigma as if they are from the
         same RV

        REF: https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
     """
    ious = np.array(ious)
    gain_mu_arr = np.array(gain_mu_arr)
    gain_std_arr = np.array(gain_std_arr)

    iou = np.mean(ious, axis=axis)
    mean_gain = np.mean(gain_mu_arr, axis=axis)

    n = gain_mu_arr.shape[0]

    # Two RVs, X and Y
    # Given mu_x, mu_y, sigma_x, sigma_y
    # sigma (standard deviation) of X + Y = np.sqrt(sigma_x**2 + sigma_y**2)
    # This gives the standard deviation of the sum, of X+Y, to get the average variance if all samples were from same
    # RV, just average the summed variance. Then sqrt it to get avg std
    sum_var = np.sum(gain_std_arr ** 2, axis=axis)
    avg_var = sum_var / n
    std_gain = np.sqrt(avg_var)

    return iou, mean_gain, std_gain


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
    plt.ylim(bottom=0)
    if f_title is not None:
        plt.title("{}".format(f_title))
    f.savefig(os.path.join(store_dir, 'gain_vs_len_{}.jpg'.format(f_name)), format='jpg')
    plt.close()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10
    results_dir = './results/test'

    # Model
    # -----

    # Base Model
    # net = new_piech_models.ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=8)
    # saved_model = './results/new_model/ContourIntegrationCSI_20191214_183159_base/best_accuracy.pth'

    # Model trained with 5 iterations
    net = new_piech_models.ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    saved_model = './results/num_iteration_explore_fix_and_sigmoid_gate/' \
                  'n_iters_5/ContourIntegrationCSI_20191208_194050/best_accuracy.pth'

    # # Without batch normalization. Dont forget to tweak the model
    # net = new_piech_models.ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # saved_model = './results/analyze_lr_rate_alexnet_bias/lr_3e-05/ContourIntegrationCSI_20191224_050603/' \
    #               'best_accuracy.pth'

    # # Control Model
    # net = new_control_models.ControlMatchParametersModel(lateral_e_size=15, lateral_i_size=15)
    # saved_model = './results/new_model/ControlMatchParametersModel_20191216_201344/best_accuracy.pth'

    # -------------------------------
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # Register Callback
    net.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # Results Folder and sub-directories
    results_store_dir = os.path.join(results_dir, os.path.dirname(saved_model).split('/')[-1])
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    gain_vs_len_dir = os.path.join(results_store_dir, 'individual_neuron_gain_vs_len')
    if not os.path.exists(gain_vs_len_dir):
        os.makedirs(gain_vs_len_dir)

    iou_vs_len_dir = os.path.join(results_store_dir, 'individual_neuron_iou_vs_len')
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
    print("====> Starting main routine")

    contour_len_arr = [1, 3, 5, 7, 9]
    beta_rotation_arr = [0]

    # gain_arr = [n_channels x len(contour_len_arr)] = [64 x 5]
    # ---------------------------------------------------------
    tgt_n_mean_gain_arr = []
    tgt_n_std_gain_arr = []

    pre_bn_max_act_n_mean_gain_arr = []
    pre_bn_max_act_n_std_gain_arr = []

    layer_in_max_act_mean_gain_arr = []
    layer_in_max_act_std_gain_arr = []

    iou_gain_vs_len_arr = []

    for gp_idx, gabor_params in enumerate(meta_data['g_params_list']):
        print(" {0} Param Set {1} {0}".format('*' * 20, gp_idx))

        tgt_n = gp_idx
        max_active_n = gabor_params[0]['extra_info']['max_active_neuron_idx']
        # Stored in the dataset meta data. The stimulus is the one the target neuron max reacted to.
        # However they may be other neurons that are more active. Max of which is max_active_n

        # Stored meta-data fields. (for coding only)
        # extra_info = {
        #     'optim_stim_act_value'
        #     'optim_stim_base_gabor_set'
        #     'optim_stim_act_orient'
        #     'max_active_neuron_is_target',
        #     'max_active_neuron_value',
        #     'max_active_neuron_idx'
        #     'orient_tuning_curve_x'
        #     'orient_tuning_curve_y'
        # }

        # (1) Get performance of target neuron
        iou_results, mean_gains, std_gains, max_act_layer_in = get_contour_gain_vs_length(
            net,
            contour_len_arr,
            beta_rotation_arr,
            data_dir,
            gp_idx,
            full_tile_size=bg_tile_size,
            n_idx=tgt_n
        )

        tgt_n_mean_gain_arr.append(mean_gains)
        tgt_n_std_gain_arr.append(std_gains)
        iou_gain_vs_len_arr.append(iou_results)

        # (2) Get performance of max active neuron before batch normalization
        # TODO: Should really call once and collect activations of multiple neurons
        if max_active_n != tgt_n:
            _, mean_gains, std_gains, _ = get_contour_gain_vs_length(
                net,
                contour_len_arr,
                beta_rotation_arr,
                data_dir,
                gp_idx,
                full_tile_size=bg_tile_size,
                n_idx=max_active_n
            )

        pre_bn_max_act_n_mean_gain_arr.append(mean_gains)
        pre_bn_max_act_n_std_gain_arr.append(std_gains)

        # Get performance of max active at contour integration layer input
        # THere is a BN layer between the edge extract out and contour integration layer in
        if max_act_layer_in != max_active_n:
            _, mean_gains, std_gains, _ = get_contour_gain_vs_length(
                net,
                contour_len_arr,
                beta_rotation_arr,
                data_dir,
                gp_idx,
                full_tile_size=bg_tile_size,
                n_idx=max_act_layer_in
            )

        layer_in_max_act_mean_gain_arr.append(mean_gains)
        layer_in_max_act_std_gain_arr.append(std_gains)

        print("Target Neuron: {}, Pre-BN max active: {}, Contour Integration Layer in max active: {}".format(
            tgt_n, max_active_n, max_act_layer_in))

        # 1. Plot individual neuron gain vs length curves
        fig = plt.figure()
        plt.errorbar(
            contour_len_arr,
            tgt_n_mean_gain_arr[-1],
            tgt_n_std_gain_arr[-1],
            label='target_neuron_{}'.format(tgt_n)
        )
        plt.errorbar(
            contour_len_arr,
            pre_bn_max_act_n_mean_gain_arr[-1],
            pre_bn_max_act_n_std_gain_arr[-1],
            label='pre_bn_max_active_neuron_{}'.format(max_active_n)
        )
        plt.errorbar(
            contour_len_arr,
            layer_in_max_act_mean_gain_arr[-1],
            layer_in_max_act_std_gain_arr[-1],
            label='lay_in_max_active_neuron_{}'.format(max_act_layer_in)
        )

        plt.ylim(bottom=0)
        plt.legend()
        plt.ylabel("Gain")
        plt.xlabel("Contour Length")

        fig_name = 'neuron_{}'.format(tgt_n)
        fig.savefig(os.path.join(gain_vs_len_dir, fig_name + '.jpg'))

        # import pdb
        # pdb.set_trace()

        plt.close()
        plot_iou_vs_contour_length(contour_len_arr, iou_results, iou_vs_len_dir, fig_name)

        # TODO: Save tuning curves
        # plt.figure()
        # plt.plot(
        #     gabor_params[0]['extra_info']['orient_tuning_curve_x'],
        #     gabor_params[0]['extra_info']['orient_tuning_curve_y']
        # )
        # plt.title("Target Neuron Tuning Tuning Curve")

    # -----------------------------------------------------------------------------------
    # Population Results (n_channels, c_len_arr).
    # Avg over images already done
    # -----------------------------------------------------------------------------------
    tgt_n_mean_gain_arr = np.array(tgt_n_mean_gain_arr)
    tgt_n_std_gain_arr = np.array(tgt_n_std_gain_arr)

    pre_bn_max_act_n_mean_gain_arr = np.array(pre_bn_max_act_n_mean_gain_arr)
    pre_bn_max_act_n_std_gain_arr = np.array(pre_bn_max_act_n_std_gain_arr)

    layer_in_max_act_mean_gain_arr = np.array(layer_in_max_act_mean_gain_arr)
    layer_in_max_act_std_gain_arr = np.array(layer_in_max_act_std_gain_arr)

    # Average over all Neurons
    # ------------------------
    all_iou, all_gain_mu, all_gain_sigma = get_averaged_results(
        iou_gain_vs_len_arr,
        tgt_n_mean_gain_arr,
        tgt_n_std_gain_arr
    )

    fig_name = 'average_all_neurons'
    title = 'Average over all neurons'
    plot_gain_vs_contour_length(
        contour_len_arr,
        all_gain_mu,
        all_gain_sigma,
        results_store_dir,
        fig_name,
        title
    )

    pre_bn_all_iou, pre_bn_all_gain_mu, pre_bn_all_gain_sigma = get_averaged_results(
        iou_gain_vs_len_arr,
        pre_bn_max_act_n_mean_gain_arr,
        pre_bn_max_act_n_std_gain_arr
    )

    fig_name = 'max_active_pre_batch_normalization'
    title = 'All neurons max active for stimulus before batch normalization'
    plot_gain_vs_contour_length(
        contour_len_arr,
        pre_bn_all_gain_mu,
        pre_bn_all_gain_sigma,
        results_store_dir,
        fig_name,
        title
    )

    layer_in_all_iou, layer_in_all_gain_mu, layer_in_all_gain_sigma = get_averaged_results(
        iou_gain_vs_len_arr,
        layer_in_max_act_mean_gain_arr,
        layer_in_max_act_std_gain_arr
    )

    fig_name = 'max_active_layer_in'
    title = 'All neurons max active for stimulus at contour integration layer input'
    plot_gain_vs_contour_length(
        contour_len_arr,
        layer_in_all_gain_mu,
        layer_in_all_gain_sigma,
        results_store_dir,
        fig_name,
        title
    )

    import pdb
    pdb.set_trace()
