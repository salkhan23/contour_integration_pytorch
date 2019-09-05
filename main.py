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
from models.piech_models import CurrentSubtractiveInhibition, CurrentDivisiveInhibition
import models.control_models as control_models


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    train_batch_size = 16
    test_batch_size = 1
    learning_rate = 0.001
    num_epochs = 50
    random_seed = 10

    results_store_dir = './results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
    # model = CurrentSubtractiveInhibition().to(device)
    model = CurrentDivisiveInhibition().to(device)
    # model = control_models.CmMatchIterations().to(device)
    # model = control_models.CmMatchParameters().to(device)
    # model = control_models.CmClassificationHeadOnly().to(device)

    # print(model)
    print("Name: {}".format(model.__class__.__name__))
    print("Classifier Head: {}".format(model.post.__class__.__name__))

    results_store_dir = os.path.join(
        results_store_dir,
        model.__class__.__name__ + datetime.now().strftime("_%Y%m%d_%H%M%S"))
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_set_dir = "./data/bw_gabors_10_frag_fullTile_32_fragTile_20"
    print("Source: {}".format(data_set_dir))

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
        """ Train for one Epoch  over the train data set """
        model.train()
        e_loss = 0
        e_iou = 0

        for iteration, (img, label) in enumerate(train_data_loader, 1):
            optimizer.zero_grad()  # zero the parameter gradients

            img = img.to(device)
            label = label.to(device)

            label_out = model(img)
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

                label_out = model(img)
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
    training_start_time = datetime.now()

    detect_thres = 0.5

    train_history = []
    val_history = []
    lr_history = []

    for epoch in range(num_epochs):

        epoch_start_time = datetime.now()

        train_history.append(train())
        val_history.append(validate())

        lr_history.append(get_lr(optimizer))

        print("Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, IoU={:0.4f}. Time {}".format(
            epoch, num_epochs,
            train_history[epoch][0],
            train_history[epoch][1],
            val_history[epoch][0],
            val_history[epoch][1],
            datetime.now() - epoch_start_time
        ))

    print('Finished Training. Training took {}'.format(datetime.now() - training_start_time))

    torch.save(
        model.state_dict(),
        os.path.join(results_store_dir, 'trained_epochs_{}.pth'.format(num_epochs))
    )

    # Write results summary file
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    with open(summary_file, 'w+') as handle:
        handle.write("Data Set         : {}\n".format(data_set_dir))
        handle.write("Train images     : {}\n".format(len(train_set.images)))
        handle.write("Val images       : {}\n".format(len(val_set.images)))
        handle.write("Train batch size : {}\n".format(train_batch_size))
        handle.write("Val batch size   : {}\n".format(test_batch_size))
        handle.write("Epochs           : {}\n".format(num_epochs))
        handle.write("Model Name       : {}\n".format(model.__class__.__name__))
        handle.write("Classifier Head  : {}\n".format(model.post.__class__.__name__))
        handle.write("Lateral Excitatory Connections Size: {}\n".format(model.lateral_e.weight.shape))
        handle.write("Lateral Inhibitory Connections Size: {}\n".format(model.lateral_i.weight.shape))
        handle.write("{}\n".format('-'*80))

        handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
        handle.write("learning rate    : {}\n".format(learning_rate))
        handle.write("Loss             : {}\n".format(criterion.__class__.__name__))
        handle.write("{}\n".format('-' * 80))

        handle.write("IoU Threshold    : {}\n".format(detect_thres))
        handle.write("{}\n".format('-' * 80))

        handle.write("Training details\n")
        handle.write("Epoch, train_loss, train_iou, val_loss, val_iou, lr\n")
        for e_idx in range(num_epochs):
            handle.write("[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {}],\n".format(
                e_idx,
                train_history[e_idx][0],
                train_history[e_idx][1],
                val_history[e_idx][0],
                val_history[e_idx][1],
                lr_history[e_idx]
            ))

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
    # input("Press any key to continue")
