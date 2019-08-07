# ------------------------------------------------------------------------------------
# Piech - 2013 - Current Based Model with Subtractive Inhibition
# ------------------------------------------------------------------------------------
from datetime import datetime
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset
import fields1993_stimuli
from models.cont_int_model import CurrentSubtractiveInhibition
import utils
from models.control_model import ControlModel


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()

    train_batch_size = 16
    test_batch_size = 1
    device = torch.device("cuda")
    learning_rate = 0.001
    num_epochs = 10

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("Loading Model")
    model = CurrentSubtractiveInhibition().to(device)
    # model = ControlModel().to(device)
    # print(model)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    print("Setting up data loaders ")
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

    train_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'train'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize
    )

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'val'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Length of the train data loader {}".format(len(train_data_loader)))

    # -----------------------------------------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss().to(device)

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    print("Starting Training ")
    epoch_start_time = datetime.now()

    # Zero the parameter gradients
    optimizer.zero_grad()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        epoch_iou = 0
        detect_thres = 0.5

        for iteration, batch in enumerate(train_data_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            label_out = model(image)

            batch_loss = criterion(label_out, label.float())

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

            predictions = (label_out > detect_thres)
            epoch_iou += utils.intersection_over_union(predictions.float(), label.float())

        print("Epoch {}, Loss = {}, IoU={}".format(
            epoch, epoch_loss / len(train_data_loader), epoch_iou / len(train_data_loader)))

    print('Finished Training. Training took {}'.format(datetime.now() - epoch_start_time))

    # --------------------------------------------------------------------------------------
    # View Predictions
    # --------------------------------------------------------------------------------------
    model.eval()
    detect_thresh = 0.5

    with torch.no_grad():
        for batch in val_data_loader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            label_out = model(image)

            label_out = label_out.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            image = image.cpu().detach().numpy()

            image = np.squeeze(image, axis=0)
            image = np.transpose(image, axes=(1, 2, 0))
            image = utils.normalize_image(image)

            label_out = np.squeeze(label_out, axis=(0, 1))
            label_out = (label_out >= detect_thresh)

            fields1993_stimuli.plot_label_on_image(
                image, label_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0))
            plt.title("Prediction")

            labeled_image = fields1993_stimuli.plot_label_on_image(
                image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0))
            plt.title("True Label")

            import pdb
            pdb.set_trace()
