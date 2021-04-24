# ---------------------------------------------------------------------------------------
# Given a Trained Model get its performance (Loss, Iou) over Validation Dataset
# Also get validation performance as a function of contour length
# ---------------------------------------------------------------------------------------
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
import utils

import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models


def get_performance(model, device_to_use, data_loader):
    """

    :param model:
    :param device_to_use:
    :param data_loader:
    :return:
    """

    criterion = nn.BCEWithLogitsLoss().to(device_to_use)
    detect_thres = 0.5

    model.eval()
    e_loss = 0
    e_iou = 0

    with torch.no_grad():
        for iteration, (img, label) in enumerate(data_loader, 1):
            img = img.to(device_to_use)
            label = label.to(device_to_use)

            label_out = model(img)
            batch_loss = criterion(label_out, label.float())

            e_loss += batch_loss.item()
            preds = torch.sigmoid(label_out) > detect_thres
            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

    e_loss = e_loss / len(data_loader)
    e_iou = e_iou / len(data_loader)

    return e_iou, e_loss


def get_full_data_set_performance(model, data_dir, device_to_use, beta_arr=None):
    """

    :param beta_arr:
    :param model:
    :param data_dir:
    :param device_to_use:
    :return:
    """
    metadata_file = os.path.join(data_dir, 'dataset_metadata.pickle')
    with open(metadata_file, 'rb') as h:
        metadata = pickle.load(h)

    # Pre-processing
    normalize = transforms.Normalize(
        mean=metadata['channel_mean'],
        std=metadata['channel_std']
    )

    data_set = dataset.Fields1993(
        data_dir=os.path.join(data_dir, 'val'),
        bg_tile_size=metadata["full_tile_size"],
        transform=normalize,
        subset_size=None,
        c_len_arr=None,
        beta_arr=beta_arr,
        alpha_arr=None,
        gabor_set_arr=None
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )

    iou, loss = get_performance(model, device_to_use, data_loader)
    print("Performance over entire data set: Loss {:0.4f}, IoU = {:0.4f}".format(loss, iou))

    return iou, loss


def get_performance_per_len(
        model, data_dir, device_to_use, c_len_arr=np.array([1, 3, 5, 7, 9]), beta_arr=None):
    """

    :param beta_arr:
    :param model:
    :param data_dir:
    :param device_to_use:
    :param c_len_arr:
    :return:
    """

    metadata_file = os.path.join(data_dir, 'dataset_metadata.pickle')
    with open(metadata_file, 'rb') as h:
        metadata = pickle.load(h)

    # Pre-processing
    normalize = transforms.Normalize(
        mean=metadata['channel_mean'],
        std=metadata['channel_std']
    )

    c_len_ious = []
    c_len_loss = []

    for c_len_idx, c_len in enumerate(c_len_arr):

        print("processing contours of length = {}".format(c_len))

        data_set = dataset.Fields1993(
            data_dir=os.path.join(data_dir, 'val'),
            bg_tile_size=metadata["full_tile_size"],
            transform=normalize,
            subset_size=None,
            c_len_arr=[c_len],
            beta_arr=beta_arr,
            alpha_arr=None,
            gabor_set_arr=None
        )

        data_loader = DataLoader(
            dataset=data_set,
            num_workers=4,
            batch_size=1,
            shuffle=True,
            pin_memory=True
        )

        iou, loss = get_performance(model, device_to_use, data_loader)
        print("Performance for c_len {}: Loss {:0.4f}, IoU = {:0.4f}".format(c_len, loss, iou))

        c_len_ious.append(iou)
        c_len_loss.append(loss)

    return c_len_ious, c_len_loss


def plot_iou_per_contour_length(c_len_arr, iou_arr, f_title="IoU vs Contour length", file_name=None):
    """
    Plot the results of get_performance_per_len.

    :param c_len_arr: contours lengths over which the results were run
    :param iou_arr: iou per len array
    :param file_name:  if provided, will store a figure in that location
    :param f_title:

    :return:
    """
    f = plt.figure()

    # PLot IoU
    plt.plot(c_len_arr, iou_arr)
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.axhline(np.mean(iou_arr), label='average_iou = {:0.2f}'.format(
        np.mean(iou_arr)), color='red', linestyle=':')
    plt.legend()
    plt.title(f_title)

    if file_name is not None:
        f.savefig(file_name)
        plt.close(f)


def plot_loss_per_contour_length(c_len_arr, loss_arr, f_title="Loss vs Contour Length", file_name=None):
    """
    Plots the results of get_performance_per_len.

    :param c_len_arr: contours lengths over which the results were run
    :param loss_arr: Plot loss oer contour length. Separate Figure
    :param file_name:  if provided, will store a figure in that location
    :param f_title:

    :return:
    """
    f = plt.figure()

    # PLot IoU
    plt.plot(c_len_arr, loss_arr)
    plt.xlabel("Contour length")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.legend()
    plt.title(f_title)

    if file_name is not None:
        f.savefig(file_name)
        plt.close(f)


if __name__ == "__main__":
    random_seed = 5

    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # model = new_piech_models.ContourIntegrationAlexnet(cont_int_layer)
    # saved_model = \
    #     'results/new_model/ContourIntegrationCSI_20200117_092743_baseline_n_iters_5_latrf_15' \
    #     '/best_accuracy.pth'

    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    # net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    # saved_model = "./results/contour_dataset_multiple_runs/" \
    #               "positive_lateral_weights_with_independent_BN_best_gain_curves/run_1" \
    #               "/best_accuracy.pth"

    cont_int_layer = new_control_models.ControlMatchParametersLayer(
         lateral_e_size=15, lateral_i_size=15)
    saved_model = './results/contour_dataset_multiple_runs/control_mp_100_epochs/random_seed_1/' \
                  'ContourIntegrationResnet50_ControlMatchParametersLayer_20210414_002232/' \
                  'best_accuracy.pth'

    data_set_dir = "./data/channel_wise_optimal_full14_frag7"

    results_dir = os.path.dirname(saved_model)
    results_dir = os.path.join(results_dir, 'validation')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # ---------------------------------------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))

    # ----------------------------------------------------------------------------
    # Straight Contours
    # ----------------------------------------------------------------------------
    print("===> Getting performance for Straight Contours")
    beta_rotations_arr = [0]
    c_length_arr = np.array([1, 3, 5, 7, 9])

    st_contours_full_iou, st_contours_full_loss = get_full_data_set_performance(
        net, data_set_dir, device_to_use=device, beta_arr=beta_rotations_arr)

    st_contours_c_len_iou_arr, st_contours_c_len_loss_arr = get_performance_per_len(
        net, data_set_dir, device_to_use=device, c_len_arr=c_length_arr, beta_arr=beta_rotations_arr)

    plot_iou_per_contour_length(
        c_length_arr,
        st_contours_c_len_iou_arr,
        f_title='Straight Contours',
        file_name=os.path.join(results_dir, 'iou_straight.png'))

    plot_loss_per_contour_length(
        c_length_arr, st_contours_c_len_loss_arr, f_title='Straight Contours')

    # ----------------------------------------------------------------------------
    # Curved Contours
    # ----------------------------------------------------------------------------
    print("===> Getting performance for Curved Contours")
    beta_rotations_arr = [15]
    c_length_arr = np.array([1, 3, 5, 7, 9])

    curve_contours_full_iou, curve_contours_full_loss = get_full_data_set_performance(
        net, data_set_dir, device_to_use=device, beta_arr=beta_rotations_arr)

    curve_contours_c_len_iou_arr, curve_contours_c_len_loss_arr = get_performance_per_len(
        net, data_set_dir, device_to_use=device, c_len_arr=c_length_arr, beta_arr=beta_rotations_arr)

    plot_iou_per_contour_length(
        c_length_arr,
        curve_contours_c_len_iou_arr,
        f_title='Curved Contours',
        file_name=os.path.join(results_dir, 'iou_curved.png'))

    plot_loss_per_contour_length(
        c_length_arr, curve_contours_c_len_loss_arr, f_title='Curved Contours')

    # ----------------------------------------------------------------------------
    # Full Dataset
    # ----------------------------------------------------------------------------
    print("Getting performance for All Contours")
    beta_rotations_arr = None
    c_length_arr = np.array([1, 3, 5, 7, 9])

    full_iou, full_loss = get_full_data_set_performance(
        net, data_set_dir, device_to_use=device, beta_arr=beta_rotations_arr)

    full_c_len_iou_arr, full_c_len_arr_loss_arr = get_performance_per_len(
        net, data_set_dir, device_to_use=device, c_len_arr=c_length_arr, beta_arr=beta_rotations_arr)

    plot_iou_per_contour_length(
        c_length_arr,
        full_c_len_iou_arr,
        f_title='All Contours',
        file_name=os.path.join(results_dir, 'iou_all.png'))

    plot_loss_per_contour_length(
        c_length_arr, full_c_len_arr_loss_arr, f_title='All Contours')

    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
