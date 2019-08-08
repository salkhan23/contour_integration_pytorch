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
import utils
from models.cont_int_model import CurrentSubtractiveInhibition
from models.control_model import ControlModel


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    train_batch_size = 16
    test_batch_size = 1
    learning_rate = 0.001
    num_epochs = 20

    results_store_dir = './results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model")
    model = CurrentSubtractiveInhibition().to(device)
    # model = ControlModel().to(device)
    # print(model)

    results_store_dir = os.path.join(results_store_dir, model.__class__.__name__)
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -----------------------------------------------------------------------------------
    # Data Loader
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
        shuffle=True,
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
    #  Training Validation Routines
    # -----------------------------------------------------------------------------------
    def train():
        """ Train for one Epoch  over the train dataset """
        model.train()
        e_loss = 0
        e_iou = 0

        for iteration, (img, label) in enumerate(train_data_loader, 1):
            optimizer.zero_grad()  # zero the parameter gradients

            img = img.to(device)
            label = label.to(device)

            label_out, _ = model(img)
            batch_loss = criterion(label_out, label.float())

            batch_loss.backward()
            optimizer.step()

            e_loss += batch_loss.item()

            preds = (label_out > detect_thres)
            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(train_data_loader)
        e_iou = e_iou / len(train_data_loader)

        # print("Train Epoch {} Loss = {:0.4f}, IoU={:0.4f}".format(epoch, e_loss, e_iou))

        return e_loss, e_iou

    def validate():
        """ Get loss over validation set """
        model.eval()
        e_loss = 0
        e_iou = 0

        with torch.no_grad():
            for iteration, (img, label) in enumerate(val_data_loader, 1):
                img = img.to(device)
                label = label.to(device)

                label_out, _ = model(img)
                batch_loss = criterion(label_out, label.float())

                e_loss += batch_loss.item()
                preds = (label_out > detect_thres)
                e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(val_data_loader)
        e_iou = e_iou / len(val_data_loader)

        # print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

        return e_loss, e_iou

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("====> Starting Training ")
    epoch_start_time = datetime.now()

    detect_thres = 0.5

    train_history = []
    val_history = []

    for epoch in range(num_epochs):

        train_history.append(train())
        val_history.append(validate())

        print("Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, IoU={:0.4f}".format(
            epoch, num_epochs,
            train_history[epoch][0],
            train_history[epoch][1],
            val_history[epoch][0],
            val_history[epoch][1]
        ))

    print('Finished Training. Training took {}'.format(datetime.now() - epoch_start_time))

    torch.save(
        model.state_dict(),
        os.path.join(results_store_dir, 'trained_epochs_{}.pth'.format(num_epochs))
    )

    # -----------------------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------------------
    train_history = np.array(train_history)
    val_history = np.array(val_history)

    f = plt.figure()
    plt.title("Loss")
    plt.plot(train_history[:, 0], label='train')
    plt.plot(val_history[:, 0], label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    f.savefig(os.path.join(results_store_dir, 'loss.jpg'), format='jpg')

    f = plt.figure()
    plt.title("IoU")
    plt.plot(train_history[:, 1], label='train')
    plt.plot(val_history[:, 1], label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    f.savefig(os.path.join(results_store_dir, 'iou.jpg'), format='jpg')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    input("Press any key to continue")

    # # --------------------------------------------------------------------------------------
    # # View Predictions
    # # --------------------------------------------------------------------------------------
    # model.eval()
    # detect_thresh = 0.5
    #
    # with torch.no_grad():
    #     for batch in val_data_loader:
    #         image, label = batch
    #         image = image.to(device)
    #         label = label.to(device)
    #
    #         label_out, iter_out_arr = model(image)
    #
    #         label_out = label_out.cpu().detach().numpy()
    #         label = label.cpu().detach().numpy()
    #         image = image.cpu().detach().numpy()
    #
    #         image = np.squeeze(image, axis=0)
    #         image = np.transpose(image, axes=(1, 2, 0))
    #         image = utils.normalize_image(image)
    #
    #         label_out = np.squeeze(label_out, axis=(0, 1))
    #         label_out = (label_out >= detect_thresh)
    #
    #         for i_idx, iter_out in enumerate(iter_out_arr):
    #
    #             iter_out = iter_out.cpu().detach().numpy()
    #             iter_out = np.squeeze(iter_out, axis=(0, 1))
    #             iter_out = (iter_out >= detect_thresh)
    #
    #             labeled_image = fields1993_stimuli.plot_label_on_image(
    #                 image, iter_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0), display_figure=False)
    #
    #             labeled_image = fields1993_stimuli.plot_label_on_image(
    #                  labeled_image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0))
    #             plt.title("Iteration {}".format(i_idx))
    #
    #
    #
    #         # fields1993_stimuli.plot_label_on_image(
    #         #     image, label_out, f_tile_size=val_set.bg_tile_size, edge_color=(0, 255, 0))
    #         # plt.title("Prediction")
    #         #
    #         # labeled_image = fields1993_stimuli.plot_label_on_image(
    #         #     image, label, f_tile_size=val_set.bg_tile_size, edge_color=(255, 0, 0))
    #         # plt.title("True Label")
    #
    #             import pdb
    #             pdb.set_trace()
