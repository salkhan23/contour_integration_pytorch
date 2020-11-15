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


def get_performance_per_len(model, data_dir, device_to_use, c_len_arr=np.array([1, 3, 5, 7, 9]), beta_arr=None):
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


if __name__ == "__main__":
    random_seed = 5

    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # model = new_piech_models.ContourIntegrationAlexnet(cont_int_layer)
    # saved_model = \
    #     'results/new_model/ContourIntegrationCSI_20200117_092743_baseline_n_iters_5_latrf_15' \
    #     '/best_accuracy.pth'

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = "./results/new_model_resnet_based/" \
                  "ContourIntegrationResnet50_CurrentSubtractInhibitLayer_run_1_20200924_183734" \
                  "/best_accuracy.pth"

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
    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

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

    plt.figure('iou_fig')
    plt.plot(c_length_arr, st_contours_c_len_iou_arr, label='straight contours')
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid()
    plt.ylim([0, 1])
    plt.title("IoU vs Length")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'iou_straight.png'))

    plt.figure('loss_fig')
    plt.plot(c_length_arr, st_contours_c_len_loss_arr, label='straight contours')
    plt.grid()
    plt.xlabel("Contour length")
    plt.ylabel("Loss")
    plt.title("Loss vs Length")
    plt.legend()

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

    plt.figure('iou_fig')
    plt.plot(c_length_arr, curve_contours_c_len_iou_arr, label='curved contours')
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid()
    plt.ylim([0, 1])
    plt.title("IoU vs Length")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'iou_curved.png'))

    plt.figure('loss_fig')
    plt.plot(c_length_arr, curve_contours_c_len_loss_arr, label='curved contours')
    plt.grid()
    plt.xlabel("Contour length")
    plt.ylabel("Loss")
    plt.title("Loss vs Length")
    plt.legend()

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

    plt.figure('iou_fig')
    plt.plot(c_length_arr, full_c_len_iou_arr, label='all')
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid()
    plt.ylim([0, 1])
    plt.title("IoU vs Length")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'iou_full.png'))

    plt.figure('loss_fig')
    plt.plot(c_length_arr, full_c_len_arr_loss_arr, label='all')
    plt.grid()
    plt.xlabel("Contour length")
    plt.ylabel("Loss")
    plt.title("Loss vs Length")
    plt.legend()

    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
