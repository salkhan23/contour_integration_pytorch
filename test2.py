# ---------------------------------------------------------------------------------------
# Contour Gain/Spacing variant for natural images
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import heapq

import torch
from torchvision import transforms

import models.new_piech_models as new_piech_models
from generate_pathfinder_dataset import OnlineNaturalImagesPathfinder
from torch.utils.data import DataLoader
import utils


edge_extract_act = []
cont_int_in_act = []
cont_int_out_act = []


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at Edge Extract layer
        Callback to Retrieve the activations output of edge Extract layer
    """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer
        Callback to Retrieve the activations input & output of the contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


def process_image(model, devise_to_use, ch_mus, ch_sigmas, in_img):
    """
    Pass image through model and get iou score of the prediction if in_img_label is not None

    :param model:
    :param devise_to_use:
    :param in_img:
    :param ch_mus:
    :param ch_sigmas:
    :return:
    """
    # Zero all collected variables
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    edge_extract_act = 0
    cont_int_in_act = 0
    cont_int_out_act = 0

    normalize = transforms.Normalize(mean=ch_mus, std=ch_sigmas)
    model_in_img = normalize(in_img.squeeze())
    model_in_img = model_in_img.to(devise_to_use).unsqueeze(0)

    # Pass the image through the model
    model.eval()
    if isinstance(model, new_piech_models.JointPathfinderContourResnet50):
        # Output is contour_dataset_out, pathfinder_out
        _, label_out = model(model_in_img)
    else:
        label_out = model(model_in_img)

    label_out = torch.sigmoid(label_out)
    return label_out


def plot_channel_responses(model, img, ch_idx, dev, ch_mean, ch_std, item=None):
    """" Debugging plot"""
    f, ax_arr = plt.subplots(1, 3, figsize=(21, 7))

    label_out = process_image(model, dev, ch_mean, ch_std, img)

    img = np.transpose(img.squeeze(), axes=(1, 2, 0))
    ax_arr[0].imshow(img)

    tgt_ch_in_acts = cont_int_in_act[0, ch_idx, :, :]
    ax_arr[1].imshow(tgt_ch_in_acts)
    ax_arr[1].set_title("In.")
    if item:
        ax_arr[2].scatter(item.position[1], item.position[0], marker='o', color='magenta', s=120)
        ax_arr[2].set_title("In.\n @ {} ={:0.4f}[stored]".format(
            item.position, tgt_ch_in_acts[item.position[0], item.position[1]]))

    tgt_ch_out_acts = cont_int_out_act[0, ch_idx, :, :]
    max_act_idx = np.argmax(tgt_ch_out_acts)  # 1d index
    max_act_idx = np.unravel_index(max_act_idx, tgt_ch_out_acts.shape)  # 2d i
    ax_arr[2].imshow(tgt_ch_out_acts)
    ax_arr[2].set_title("Out\n. Max = {:.4f} @ {}".format(np.max(tgt_ch_out_acts), max_act_idx))
    ax_arr[2].scatter(max_act_idx[1], max_act_idx[0], marker='+', color='red', s=120)

    if item:
        ax_arr[2].scatter(item.position[1], item.position[0], marker='o', color='magenta', s=120)
        ax_arr[2].set_title("Out.\n @ {} ={:0.4f} [Current max] \n @ {} ={:0.4f}[stored]".format(
            max_act_idx, np.max(tgt_ch_out_acts), item.position,
            tgt_ch_out_acts[item.position[0], item.position[1]]))
    #     ax_arr[2].scatter(item.ep1[1] // 4, item.ep1[0] // 4, marker='o', color='magenta', s=60)
    #     ax_arr[2].scatter(item.ep2[1] // 4, item.ep2[0] // 4, marker='o', color='magenta', s=60)
    #     ax_arr[2].scatter(item.c1[:, 1] // 4, item.c1[:, 0] // 4, marker='.', color='magenta')

    title = "Prediction {:0.4f}".format(label_out.item())
    if item:
        title = \
            "GT={}, prediction={:0.4f}, Stored  prediction {:0.4f}, Out max act = {:0.4f}," \
            " position {}".format(
                item.gt, label_out.item(), item.prediction, item.activation, item.position)

    f.suptitle(title)


def get_filtered_results(in_responses, out_responses, n, valid_meas_thresh=-1000):

    valid_resp_idxs = np.nonzero(out_responses > valid_meas_thresh)

    r_out = out_responses[valid_resp_idxs]
    r_in = in_responses[valid_resp_idxs]

    idxs = r_out.argsort()
    r_out = r_out[idxs]
    r_in = r_in[idxs]

    r_out = r_out[::-1]  # highest first
    r_in = r_in[::-1]

    if len(r_out) > n:
        r_out = r_out[:n]
        r_in = r_in[:n]

    # print("Number of samples ={}, \nr in {}. \nr out ={}".format(len(r_out), r_in, r_out))

    return r_in, r_out


def main(model, base_results_dir):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    n_channels = 64
    top_n = 50

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    biped_dataset_dir = './data/BIPED/edges'
    biped_dataset_type = 'train'
    n_biped_imgs = 1000

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    contour_lengths_bins = [20, 50, 100, 150, 200]

    mean_out_per_len_mat = np.zeros((len(contour_lengths_bins), n_channels))
    std_out_per_len_mat = np.zeros_like(mean_out_per_len_mat)

    mean_in_per_len_mat = np.zeros((len(contour_lengths_bins), n_channels))
    std_in_per_len_mat = np.zeros_like(mean_out_per_len_mat)

    results_dir = os.path.join(base_results_dir, 'experiment_gain_vs_len_natural_images')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    idv_channels_results_dir = os.path.join(results_dir, 'individual_channels')
    if not os.path.exists(idv_channels_results_dir):
        os.makedirs(idv_channels_results_dir)

    summary_file = os.path.join(results_dir, 'summary.txt')
    f_handle = open(summary_file, 'w+')

    for bin_idx, bin_len in enumerate(contour_lengths_bins):

        min_len = bin_len
        max_len = 500
        bin_start_time = datetime.now()

        if bin_idx < len(contour_lengths_bins) - 1:
            max_len = contour_lengths_bins[bin_idx + 1]

        print("Generating Images/labels with contour of length in [{}, {}]".format(
            min_len, max_len - 1))

        data_set = OnlineNaturalImagesPathfinder(
            data_dir=biped_dataset_dir,
            dataset_type=biped_dataset_type,
            transform=None,
            subset_size=n_biped_imgs,
            resize_size=(256, 256),
            min_contour_len=min_len,
            max_contour_len=max_len,
            p_connect=1
        )

        data_loader = DataLoader(
            dataset=data_set,
            num_workers=0,
            batch_size=1,  # Has to be 1, returned contours are of different sizes
            shuffle=False,
            pin_memory=True
        )

        per_len_in_responses = []
        per_len_out_responses = []

        for iteration, data_loader_out in enumerate(data_loader, 1):

            print("Iteration {}".format(iteration))

            if data_loader_out[0].dim() == 4:  # if valid image

                img, label, sep_c_label, full_label, d, org_img_idx, \
                    c1, c2, start_point, end_point = data_loader_out

                label_out = process_image(model, dev, ch_mean, ch_std, img)

                # Remove batch dimension
                label = np.squeeze(label)
                # sep_c_label = np.squeeze(sep_c_label)
                # full_label = np.squeeze(full_label)
                # d = np.squeeze(d)
                org_img_idx = np.squeeze(org_img_idx)
                c1 = np.squeeze(c1)
                c2 = np.squeeze(c2)
                start_point = np.squeeze(start_point)
                end_point = np.squeeze(end_point)

                if label:  # only consider connected samples

                    max_out_channel_responses = np.ones(n_channels) * -10000
                    max_in_channel_responses = np.ones(n_channels) * -10000

                    for ch_idx in range(n_channels):

                        # Target channel activation
                        curr_tgt_ch_acts = cont_int_out_act[0, ch_idx, :, :]
                        curr_max_act = np.max(curr_tgt_ch_acts)

                        curr_max_act_idx = np.argmax(curr_tgt_ch_acts)  # 1d index
                        curr_max_act_idx = np.unravel_index(
                            curr_max_act_idx, curr_tgt_ch_acts.shape)  # 2d idx

                        # Check for valid sample:
                        # 1. Endpoints should be connected
                        # 2. max_active should be at most one pixel away from the contour
                        min_d_to_contour = np.min(
                            data_set.get_distance_point_and_contour(curr_max_act_idx, c1 // 4))

                        if min_d_to_contour < 1.5:
                            max_out_channel_responses[ch_idx] = curr_max_act
                            max_in_channel_responses[ch_idx] = \
                                cont_int_in_act[
                                    0, ch_idx, curr_max_act_idx[0], curr_max_act_idx[1]]

                    per_len_out_responses.append(max_out_channel_responses)
                    per_len_in_responses.append(max_in_channel_responses)

        # -------------------------------------------------------------------------------
        # Process the Results
        # -------------------------------------------------------------------------------
        per_len_out_responses = np.array(per_len_out_responses)
        per_len_in_responses = np.array(per_len_in_responses)

        top_n = 10

        for ch_idx in range(n_channels):

            ch_out_resps, ch_in_resps = get_filtered_results(
                per_len_in_responses[:, ch_idx],
                per_len_out_responses[:, ch_idx],
                top_n
            )

            mean_out_per_len_mat[bin_idx, ch_idx] = np.mean(ch_out_resps)
            std_out_per_len_mat[bin_idx, ch_idx] = np.std(ch_out_resps)

            mean_in_per_len_mat[bin_idx, ch_idx] = np.mean(ch_in_resps)
            std_in_per_len_mat[bin_idx, ch_idx] = np.std(ch_in_resps)

    # ---------------------------------------------------------------------------------
    #  After all results are collected print them in the summary file
    # ---------------------------------------------------------------------------------
    # Saves the results
    np.set_printoptions(precision=3)
    f_handle.write("Inputs Means: \n")
    print(mean_in_per_len_mat, file=f_handle)
    f_handle.write('\n')

    f_handle.write("Inputs STD: \n")
    print(std_in_per_len_mat, file=f_handle)
    f_handle.write('\n')

    f_handle.write("Outputs Means: \n")
    print(mean_out_per_len_mat, file=f_handle)
    f_handle.write('\n')

    f_handle.write("Outputs STD: \n")
    print(std_out_per_len_mat, file=f_handle)
    f_handle.write('\n')

    f_handle.close()

    for ch_idx in range(n_channels):
        f, ax = plt.subplots()

        ax.errorbar(
            contour_lengths_bins,
            mean_in_per_len_mat[:, ch_idx],
            std_in_per_len_mat[:, ch_idx],
            label='In act')

        ax.errorbar(
            contour_lengths_bins,
            mean_out_per_len_mat[:, ch_idx],
            std_out_per_len_mat[:, ch_idx],
            label='Out act')

        ax.set_title("Activations. Channel {}".format(ch_idx))
        ax.grid()
        ax.legend()
        ax.set_xlabel("Lengths")
        ax.set_ylabel("Activations")
        f.savefig(os.path.join(
            idv_channels_results_dir, 'activations_channel_{}.png'.format(ch_idx)))

        plt.close(f)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7

    # Immutable
    # ---------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    plt.ion()
    start_time = datetime.now()

    # Model
    # ------
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # net = new_piech_models.JointPathfinderContourResnet50(cont_int_layer)
    # saved_model = \
    #     'results/joint_training/' \
    #     'JointPathfinderContourResnet50_CurrentSubtractInhibitLayer_20200719_104417_base/' \
    #     'last_epoch.pth'

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    saved_model = \
        './results/pathfinder/' \
        'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20200726_214351_puncture_various/' \
        'best_accuracy.pth'

    results_store_dir = os.path.dirname(saved_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))
    main(net, results_store_dir)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()