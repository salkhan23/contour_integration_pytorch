# ---------------------------------------------------------------------------------------
#  [Test 1] As the number of co-aligned fragments increase, both contour integration and
#  the probability of detection increases.
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.piech_models as piech_models
from torchvision import transforms
import dataset
import utils
import fields1993_stimuli

edge_extract_act = 0
contour_integration_in_act = 0
contour_integration_out_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at the output of conv1"""
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_in_cb(self, layer_in, layer_out):
    """ Attach at the output of the bn1 layer. After this a relu is done before it is fed into cont int layer """
    global contour_integration_in_act
    contour_integration_in_act = nn.functional.relu(layer_out).cpu().detach().numpy()


def contour_integration_out_cb(self, layer_in, layer_out):
    """ Attach at post layer. Its input is the output of the contour integration layer"""
    global contour_integration_out_act
    contour_integration_out_act = layer_in[0].cpu().detach().numpy()


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


def get_center_neuron_acts(model, d_loader):

    model.eval()

    # Attach activation hooks
    model.conv1.register_forward_hook(edge_extract_cb)
    model.bn1.register_forward_hook(contour_integration_in_cb)
    model.post.conv1.register_forward_hook(contour_integration_out_cb)

    center_neuron_gains_arr = np.zeros((len(d_loader), 64))

    # fig_center_neuron, center_ax_arr = plt.subplots(3, 1, num='Center neurons')
    # fig_center_neuron.suptitle("Center Neuron All channels")

    with torch.no_grad():
        for iteration, (img, label) in enumerate(d_loader, 0):
            img = img.to(device)
            label = label.to(device)

            label_out = model(img)

            label_out = label_out.cpu().detach().numpy()
            label_out = label_out.astype(int)

            label = label.cpu().detach().numpy()
            img = img.cpu().detach().numpy()
            img = np.squeeze(img, axis=0)
            img = np.transpose(img, axes=(1, 2, 0))

            print("Label Difference: {}".format((label_out.astype(int) - label).sum()))

            # # Display Image & Labels
            # display_img = (img - img.min()) / (img.max() - img.min()) * 255.
            # # plt.imshow(display_img.astype('uint8'))
            #
            # labeled_img = fields1993_stimuli.plot_label_on_image(
            #     display_img, label, np.array([32, 32]), edge_color=(255, 0, 0), edge_width=3, display_figure=False)
            # fields1993_stimuli.plot_label_on_image(
            #     labeled_img, label_out, np.array([32, 32]), edge_color=(0, 255, 0), edge_width=3, display_figure=True)
            # import pdb
            # pdb.set_trace()

            # # Plot Summed Activations
            # # ------------------------
            # summed_edge_out = edge_extract_act[0, ].sum(axis=0)
            # summed_cont_int_in = contour_integration_in_act[0, ].sum(axis=0)
            # summed_cont_int_out = contour_integration_out_act[0, ].sum(axis=0)
            #
            # fig_sum_act, sum_act_ax_arr = plt.subplots(1, 3, sharey=True, squeeze=True, figsize=(18, 6))
            # fig_sum_act.suptitle("Summed Activations")
            #
            # p0 = sum_act_ax_arr[0].imshow(summed_edge_out, cmap='seismic')
            # sum_act_ax_arr[0].set_title('Edge Image')
            # plt.colorbar(p0, ax=sum_act_ax_arr[0], orientation='horizontal')
            #
            # p1 = sum_act_ax_arr[1].imshow(summed_cont_int_in, cmap='seismic')
            # sum_act_ax_arr[1].set_title("Cont Int In (BN+Relu)")
            # plt.colorbar(p1, ax=sum_act_ax_arr[1], orientation='horizontal')
            #
            # p2 = sum_act_ax_arr[2].imshow(summed_cont_int_out, cmap='seismic')
            # sum_act_ax_arr[2].set_title("Cont Int Out")
            # plt.colorbar(p2, ax=sum_act_ax_arr[2], orientation='horizontal')
            # import pdb
            # pdb.set_trace()

            # ---------------------------------------------------------------------------
            # Center Neuron Activations
            # ---------------------------------------------------------------------------
            center_neuron_edge_out = edge_extract_act[0, :, 127 // 2 + 1, 127 // 2 + 1]
            center_neuron_cont_int_in = contour_integration_in_act[0, :, 127 // 2 + 1, 127 // 2 + 1]
            center_neuron_cont_int_out = contour_integration_out_act[0, :, 127 // 2 + 1, 127 // 2 + 1]

            # # Plot Center Neuron Gains
            # # ------------------------
            # max_act_idx = np.argmax(center_neuron_edge_out)
            # center_ax_arr[0].plot(center_neuron_edge_out)
            # center_ax_arr[0].set_title("Edge Out. Max Active: Idx {}, Value={:0.2f}".format(
            #      max_act_idx, center_neuron_edge_out[max_act_idx]))
            #
            # max_act_idx = np.argmax(center_neuron_cont_int_in)
            # center_ax_arr[1].plot(center_neuron_cont_int_in)
            # center_ax_arr[1].set_title("Contour Integration In . Max Active: Idx {}, Value={:0.2f}".format(
            #     max_act_idx, center_neuron_cont_int_in[max_act_idx]))
            #
            # max_act_idx = np.argmax(center_neuron_cont_int_out)
            # center_ax_arr[2].plot(center_neuron_cont_int_out)
            # center_ax_arr[2].set_title("Contour Integration Out. Max Active: Idx {}, Value={:0.2f}".format(
            #      max_act_idx, center_neuron_cont_int_out[max_act_idx]))
            # import pdb
            # pdb.set_trace()

            center_neuron_gains = center_neuron_cont_int_out / (center_neuron_cont_int_in + 0.00001)
            center_neuron_gains_arr[iteration, ] = center_neuron_gains

    return center_neuron_gains_arr


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
    net = piech_models.CurrentSubtractiveInhibition()
    saved_model = './results/CurrentSubtractiveInhibition_20190916_081745/trained_epochs_50.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # ---------------------------------------------------------------------------------
    # Data Loaders
    # ---------------------------------------------------------------------------------
    print("====> Setting up Data Loaders")

    data_set_dir = "./data/bw_gabors_1_frag_fullTile_32_fragTile_20_centered"
    print("Source: {}".format(data_set_dir))

    # get mean/std of dataset
    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)
    print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    # Pre-processing
    normalize = transforms.Normalize(
        mean=meta_data['channel_mean'],
        std=meta_data['channel_std']
    )

    data_dir = os.path.join(data_set_dir, 'val')
    bg_tile_size = meta_data["full_tile_size"]

    c_len_arr = np.array([3, 5, 7, 9])
    iou_scores = []

    mean_gains_arr = np.zeros((len(c_len_arr), 64))  # 64 is the number of channels
    std_gains_arr = np.zeros_like(mean_gains_arr)

    for c_idx, c_len in enumerate(c_len_arr):

        print("Processing contour length = {}".format(c_len))

        data_set = dataset.Fields1993(
            data_dir=data_dir,
            bg_tile_size=bg_tile_size,
            transform=normalize,
            c_len_arr=[c_len],
            beta_arr=[0]
        )

        data_loader = DataLoader(
            dataset=data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

        # Get Performance
        test_loss, test_iou = get_performance(net, data_loader)
        print("IoU = {}".format(test_iou))
        iou_scores.append(test_iou)

        # Get Contour Gains
        gains = get_center_neuron_acts(net, data_loader)

        # Average Across all images
        gains_mean = np.mean(gains, axis=0)
        gains_std = np.std(gains, axis=0)

        mean_gains_arr[c_idx, ] = gains_mean
        std_gains_arr[c_idx, ] = gains_std

    # Contour Length vs IoU
    plt.figure("IoU vs Contour Length")
    plt.title("Intersection over Union vs Contour Length")
    plt.plot(c_len_arr, iou_scores)
    plt.xlabel("Contour Length")
    plt.ylabel("IoU Score")

    # # Contour Length vs Gain
    # f, ax_arr = plt.subplots(8, 1, sharex=True)
    # f.suptitle("Channel-wise Mean Contour Integration Gains")
    # for ch_idx in range(64):
    #     ax_arr[ch_idx % 8].plot(c_len_arr, mean_gains_arr[:, ch_idx], label='ch={}'.format(ch_idx))
    #
    # for ax in ax_arr:
    #     # ax.legend()
    #     ax.set_ylabel("Gain")
    # ax_arr[-1].set_xlabel("Contour Length")

    for ch_idx in range(64):

        if ch_idx % 8 == 0:
            f, ax_arr = plt.subplots(8, 1, sharex=True)
            f.suptitle("Channel-wise Mean Contour Integration Gains")

        ax_arr[ch_idx % 8].plot(c_len_arr, mean_gains_arr[:, ch_idx], label='ch={}'.format(ch_idx))
        ax_arr[ch_idx % 8].legend()

    ax_arr[-1].set_xlabel("Contour Length")

    import pdb
    pdb.set_trace()
