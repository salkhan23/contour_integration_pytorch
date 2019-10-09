# ---------------------------------------------------------------------------------------
# (1) As number of co-aligned fragments increase, both contour integration gain and the
#     overall detection probability increases
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.new_piech_models as new_piech_models
from torchvision import transforms
import dataset
import utils
import fields1993_stimuli

edge_extract_act = 0
contour_integration_in_act = 0
contour_integration_out_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at edge_extract layer """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer"""
    global contour_integration_in_act
    global contour_integration_out_act

    contour_integration_in_act = layer_in[0].cpu().detach().numpy()
    contour_integration_out_act = layer_out.cpu().detach().numpy()


def get_performance(model, d_loader):
    """
    TODO: device should be passed in

    :param model:
    :param d_loader:
    :return:
    """
    criterion = nn.BCEWithLogitsLoss().to(device)

    model.eval()
    detect_thresh = 0.5
    e_loss = 0
    e_iou = 0

    with torch.no_grad():
        for iteration, (img, label) in enumerate(d_loader, 1):
            img = img.to(device)
            label = label.to(device)

            label_out = model(img)
            batch_loss = criterion(label_out, label.float())

            e_loss += batch_loss.item()
            preds = (label_out > detect_thresh)

            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

    e_loss = e_loss / len(d_loader)
    e_iou = e_iou / len(d_loader)

    return e_loss, e_iou


def plot_prediction_and_label(img, label_out, label, f_tile_size):

    display_img = (img - img.min()) / (img.max() - img.min()) * 255.
    # plt.imshow(display_img.astype('uint8'))

    # Red - Ground Truth
    labeled_img = fields1993_stimuli.plot_label_on_image(
        display_img, label, f_tile_size, edge_color=(255, 0, 0), edge_width=3, display_figure=False)

    # Green - Prediction
    fields1993_stimuli.plot_label_on_image(
        labeled_img, label_out, f_tile_size, edge_color=(0, 255, 0), edge_width=3, display_figure=True)

    plt.title("Label Image. Red = Label, Green = Prediction")


def get_center_neuron_acts(model, d_loader, n_chan):

    detect_thresh = 0.5
    center_neuron_edge_act = np.zeros((len(d_loader), n_chan))
    center_neuron_cont_int_in = np.zeros_like(center_neuron_edge_act)
    center_neuron_cont_int_out = np.zeros_like(center_neuron_edge_act)

    global edge_extract_act
    global contour_integration_in_act
    global contour_integration_out_act

    # fig_center_neuron, center_ax_arr = plt.subplots(3, 1, num='Center neurons')
    # fig_center_neuron.suptitle("Center Neuron All channels")

    model.eval()
    with torch.no_grad():
        for iteration, (img, label) in enumerate(d_loader, 0):
            img = img.to(device)
            label = label.to(device)

            # Zero All activations before each iteration
            edge_extract_act = 0
            contour_integration_in_act = 0
            contour_integration_out_act = 0

            label_out = model(img)

            label_out = label_out.cpu().detach().numpy()
            label_out = (label_out > detect_thresh)
            label_out = label_out.astype(int)
            label = label.cpu().detach().numpy()

            print("Label Difference: {}".format((label_out - label).sum()))

            # Store the activations
            _, _, r, c = edge_extract_act.shape
            center_neuron_edge_act[iteration, ] = edge_extract_act[0, :, r // 2, c // 2]
            center_neuron_cont_int_in[iteration, ] = contour_integration_in_act[0, :, r // 2, c // 2]
            center_neuron_cont_int_out[iteration, ] = contour_integration_out_act[0, :, r // 2, c // 2]

            # # -------------------------------
            # # Plot Center Neuron Activations
            # # -------------------------------
            # max_idx = np.argmax(center_neuron_edge_act[iteration, ])
            # center_ax_arr[0].plot(center_neuron_edge_act[iteration, ])
            # center_ax_arr[0].set_title("Edge Out. Max Active Neuron {}, Value {:0.2f}".format(
            #     max_idx, center_neuron_edge_act[iteration, max_idx]))
            #
            # max_idx = np.argmax(center_neuron_cont_int_in[iteration, ])
            # center_ax_arr[1].plot(center_neuron_cont_int_in[iteration, ])
            # center_ax_arr[1].set_title("Contour Int In. Max Active Neuron {}, Value {:0.2f}".format(
            #     max_idx, center_neuron_cont_int_in[iteration, max_idx]))
            #
            # max_idx = np.argmax(center_neuron_cont_int_in[iteration, ])
            # center_ax_arr[1].plot(center_neuron_cont_int_in[iteration, ])
            # center_ax_arr[1].set_title("Contour Int In. Max Active Neuron {}, Value {:0.2f}".format(
            #     max_idx, center_neuron_cont_int_in[iteration, max_idx]))
            #
            # max_idx = np.argmax(center_neuron_cont_int_out[iteration, ])
            # center_ax_arr[2].plot(center_neuron_cont_int_out[iteration, ])
            # center_ax_arr[2].set_title("Contour Int Out. Max Active Neuron {}, Value {:0.2f}".format(
            #     max_idx, center_neuron_cont_int_out[iteration, max_idx]))
            #
            # import pdb
            # pdb.set_trace()

    return center_neuron_edge_act, center_neuron_cont_int_in, center_neuron_cont_int_out


def plot_contour_integration_gains(len_arr, tgt_n_idx, in_act, out_act, epsilon=1e-5, fig_ax=None, label=None):
    """

    Given a target neuron plots its contour integration gain

    :param label:
    :param fig_ax:
    :param len_arr:  array/list of contour lengths
    :param tgt_n_idx: the target neuron (channel) to get gains for
    :param in_act:  (contour len, n_images, n_channels)
    :param out_act: (contour len, n_images, n_channels)
    :param epsilon:

    :return:
    """

    gains = out_act / (in_act + epsilon)
    tgt_n_mean_gain_list = []

    for l_idx, length in enumerate(len_arr):

        tgt_n_gains = gains[l_idx, :, tgt_n_idx]
        tgt_n_mean_gain_list.append(tgt_n_gains.mean())

    if fig_ax is None:
        f, fig_ax = plt.subplots()

    fig_ax.set_title("Contour Integration Gain for Neuron {}".format(tgt_n_idx))
    fig_ax.plot(len_arr, tgt_n_mean_gain_list, marker='x', markersize=10, label=label)
    fig_ax.set_xlabel('Contour Length')
    fig_ax.set_ylabel("Gain")

    return tgt_n_mean_gain_list


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Setting up the Model ...")
    net = new_piech_models.ContourIntegrationCSI(lateral_e_size=23, lateral_i_size=23)
    saved_model = './results/new_model/ContourIntegrationCSI_20191005_195738_base/best_accuracy.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # -----------------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------------
    print("====> Setting up Data Loaders")

    data_set_dir = "./data/fitted_gabors_10_full14_frag7_centered_test"
    print("Source: {}".format(data_set_dir))

    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)

    data_dir = os.path.join(data_set_dir, 'val')
    n_images_per_set = meta_data['n_val_images_per_set']
    n_channels = 64

    bg_tile_size = meta_data["full_tile_size"]

    # ---------------------------------------------------------------------------------
    # Main Routine
    # ---------------------------------------------------------------------------------

    gabor_params_set = [9]

    print("processing Gabor Set {}".format(gabor_params_set))

    c_len_arr = np.array([1, 3, 5, 7, 9])

    iou_scores = []
    edge_extract_act_set = np.zeros((len(c_len_arr), n_images_per_set, n_channels))
    cont_int_in_act_set = np.zeros_like(edge_extract_act_set)
    cont_int_out_act_set = np.zeros_like(edge_extract_act_set)

    for c_idx, c_len in enumerate(c_len_arr):

        print("Processing contour length = {}".format(c_len))

        # # Normalization
        # if gabor_params_set is not None and len(gabor_params_set) > 1:
        #     mean = meta_data['channel_mean']
        #     std = meta_data['channel_std']
        # else:
        #     mean = meta_data['set_specific_means'][gabor_params_set[0]]
        #     std = meta_data['set_specific_std'][gabor_params_set[0]]
        mean = meta_data['channel_mean']
        std = meta_data['channel_std']

        normalize = transforms.Normalize(mean=mean, std=std)

        data_set = dataset.Fields1993(
            data_dir=data_dir,
            bg_tile_size=bg_tile_size,
            transform=normalize,
            c_len_arr=[c_len],
            beta_arr=[0],
            gabor_set_arr=gabor_params_set
        )

        data_loader = DataLoader(data_set, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

        # -------------------------------------------------------------------------------
        # Get Performance
        # -------------------------------------------------------------------------------
        test_loss, test_iou = get_performance(net, data_loader)
        iou_scores.append(test_iou)

        # -------------------------------------------------------------------------------
        # Get Center Neuron Activation
        # -------------------------------------------------------------------------------
        # Register Callbacks
        net.edge_extract.register_forward_hook(edge_extract_cb)
        net.contour_integration_layer.register_forward_hook(contour_integration_cb)

        edge_extract_act_set[c_idx, ], cont_int_in_act_set[c_idx, ], cont_int_out_act_set[c_idx, ] = \
            get_center_neuron_acts(net, data_loader, n_channels)

    # -----------------------------------------------------------------------------------
    # Plot Contour Length vs IoU
    # -----------------------------------------------------------------------------------
    plt.figure("IoU vs Contour Length")
    plt.title("Intersection over Union vs Contour Length")
    plt.plot(c_len_arr, iou_scores, marker='x', markersize=10)
    plt.xlabel("Contour Length")
    plt.ylabel("IoU Score")
    plt.ylim([0, 1.1])

    # -----------------------------------------------------------------------------------
    # Gain Plots
    # -----------------------------------------------------------------------------------
    # Most Active Neuron Contour Integration input
    avg_resp = cont_int_in_act_set.mean(axis=(0, 1))  # average across all length and all images
    max_active_contour_int_in = np.argmax(avg_resp)

    # Most Active Neuron Contour Integration Output - The responses differ per contour length
    # just get the max active for the largest contour
    avg_resp = cont_int_out_act_set[-1, :, :].mean(axis=0)
    max_active_contour_int_out = np.argmax(avg_resp)

    # Max Active Edge Extraction Neuron
    avg_resp = edge_extract_act_set.mean(axis=(0, 1))
    max_active_edge_extract = np.argmax(avg_resp)

    print("Max Active: Edge Extract {}, Contour Integration: In {}. Out {}".format(
        max_active_edge_extract, max_active_contour_int_in, max_active_contour_int_out))

    # Proper Contour Integration.
    plot_contour_integration_gains(c_len_arr, max_active_contour_int_in, cont_int_in_act_set, cont_int_out_act_set)
    plt.title("Proper Contour Integration. Neuron {} (Most active input)".format(max_active_contour_int_in),)

    # Most reactive neuron at Contour Integration layer output.
    plot_contour_integration_gains(
        c_len_arr, max_active_contour_int_out, cont_int_in_act_set, cont_int_out_act_set)
    plt.title("Neuron {}. Most active at contour integration output.".format(max_active_contour_int_out))

    import pdb
    pdb.set_trace()
