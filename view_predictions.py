# ---------------------------------------------------------------------------------------
# View Iterative predictions of a model
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import dataset
import utils
import fields1993_stimuli
from models.piech_models import CurrentSubtractiveInhibition, CurrentDivisiveInhibition
import models.control_models as control_models


if __name__ == '__main__':

    # saved_model = './results/CurrentDivisiveInhibition/trained_epochs_50.pth'
    # model = CurrentDivisiveInhibition()

    saved_model = './results/CurrentSubtractiveInhibition_20190823_182503/trained_epochs_50.pth'
    model = CurrentSubtractiveInhibition()

    # saved_model = './results/CmMatchParameters/trained_epochs_50.pth'
    # model = control_models.CmMatchParameters()

    # saved_model = './results/CmMatchIterations_20190823_175326/trained_epochs_50.pth'
    # model = control_models.CmMatchIterations()

    # ----------------
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(saved_model), strict=False)
    model = model.to(device)

    # -----------------------------------------------------------------------------------
    # Data Loaders
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_set_dir = "./data/bw_gabors_5_frag_fullTile_32_fragTile_20"

    # get mean/std of dataset
    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)
    # print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    # Pre-processing
    normalize = transforms.Normalize(
        mean=meta_data['channel_mean'],
        std=meta_data['channel_std']
    )

    val_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'val'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Get performance
    # -----------------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss().to(device)

    model.eval()
    detect_thresh = 0.5
    e_loss = 0
    e_iou = 0

    with torch.no_grad():
        for iteration, (img, label) in enumerate(val_data_loader, 1):
            img = img.to(device)
            label = label.to(device)

            label_out, _ = model(img)
            batch_loss = criterion(label_out, label.float())

            e_loss += batch_loss.item()
            preds = (label_out > detect_thresh)
            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

    e_loss = e_loss / len(val_data_loader)
    e_iou = e_iou / len(val_data_loader)

    print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

    # -----------------------------------------------------------------------------------
    # View Predictions
    # -----------------------------------------------------------------------------------
    model.eval()
    detect_thresh = 0.5

    with torch.no_grad():
        for batch in val_data_loader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            label_out, iter_out_arr = model(image)

            label_out = label_out.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            image = image.cpu().detach().numpy()

            image = np.squeeze(image, axis=0)
            image = np.transpose(image, axes=(1, 2, 0))
            image = utils.normalize_image(image) * 255.0

            label_out = np.squeeze(label_out, axis=(0, 1))
            label_out = (label_out >= detect_thresh)

            n_iter = len(iter_out_arr)
            f, ax_arr = plt.subplots(1, n_iter, sharey=True, squeeze=True, figsize=(12, 3))

            # ---------------------------------------------------------------------------
            # View Iterative Predictions
            # ---------------------------------------------------------------------------
            for i_idx, iter_out in enumerate(iter_out_arr):

                iter_out = iter_out.cpu().detach().numpy()
                iter_out = np.squeeze(iter_out, axis=(0, 1))
                iter_out = (iter_out >= detect_thresh)

                labeled_image = fields1993_stimuli.plot_label_on_image(
                    image, iter_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0),
                    display_figure=False, edge_width=3)

                labeled_image = fields1993_stimuli.plot_label_on_image(
                     labeled_image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0),
                     display_figure=False, edge_width=3)

                ax_arr[i_idx].imshow(labeled_image)
                ax_arr[i_idx].set_title(i_idx)

            f.suptitle("Red=True, Green=Prediction, Yellow=Overlap")
            plt.tight_layout()

            import pdb
            pdb.set_trace()

            plt.close()

            # # ---------------------------------------------------------------------------
            # # View Final Predictions
            # # ---------------------------------------------------------------------------
            # labeled_image = fields1993_stimuli.plot_label_on_image(
            #     image, label_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0), display_figure=False)
            #
            # fields1993_stimuli.plot_label_on_image(
            #      labeled_image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0))
            # plt.title("Final. Red = True, Green Pred, Yellow=Overlap".)
            #
            # import pdb
            # pdb.set_trace()
