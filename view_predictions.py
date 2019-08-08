# ---------------------------------------------------------------------------------------
# View Iterative perdictions of a model
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
import utils
import fields1993_stimuli
from models.cont_int_model import CurrentSubtractiveInhibition
import models.control_models as control_models


if __name__ == '__main__':

    saved_model = './results/CurrentSubtractiveInhibition/trained_epochs_50.pth'
    model = CurrentSubtractiveInhibition()

    # saved_model = './results/CmMatchParameters/trained_epochs_50.pth'
    # model = control_models.CmMatchParameters()

    # saved_model = './results/CmMatchIterations/trained_epochs_50.pth'
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
    data_set_dir = "./data/single_fragment_full"

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
            image = utils.normalize_image(image)

            label_out = np.squeeze(label_out, axis=(0, 1))
            label_out = (label_out >= detect_thresh)

            # ---------------------------------------------------------------------------
            # View Iterative Predictions
            # ---------------------------------------------------------------------------
            for i_idx, iter_out in enumerate(iter_out_arr):

                iter_out = iter_out.cpu().detach().numpy()
                iter_out = np.squeeze(iter_out, axis=(0, 1))
                iter_out = (iter_out >= detect_thresh)

                labeled_image = fields1993_stimuli.plot_label_on_image(
                    image, iter_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0), display_figure=False)

                fields1993_stimuli.plot_label_on_image(
                     labeled_image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0))
                plt.title("Iteration {}. Red=True, Green=Prediction, Yellow=Overlap".format(i_idx))

                import pdb
                pdb.set_trace()

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
