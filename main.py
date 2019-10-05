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
from models.new_piech_models import ContourIntegrationCSI
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
    num_epochs = 20
    random_seed = 10

    results_store_dir = './results/new_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
    # model = CurrentSubtractiveInhibition(lateral_e_size=7, lateral_i_size=7).to(device)
    # model = CurrentDivisiveInhibition().to(device)
    # model = control_models.CmMatchIterations().to(device)
    # model = control_models.CmMatchParameters(lateral_e_size=3, lateral_i_size=3).to(device)
    # model = control_models.CmClassificationHeadOnly().to(device)
    model = ContourIntegrationCSI(lateral_e_size=22, lateral_i_size=22).to(device)

    # print(model)
    print("Name: {}".format(model.__class__.__name__))
    print(model)

    from torchsummary import summary
    summary(model, input_size=(3,256,256))

    import pdb
    pdb.set_trace()

    results_store_dir = os.path.join(
        results_store_dir,
        model.__class__.__name__ + datetime.now().strftime("_%Y%m%d_%H%M%S"))
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_load_start_time = datetime.now()

    data_set_dir = "./data/fitted_gabors_10_full14_frag7_test"
    print("Source: {}".format(data_set_dir))

    # get mean/std of dataset
    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as file_handle:
        meta_data = pickle.load(file_handle)
    # print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    # Pre-processing
    normalize = transforms.Normalize(
        mean=meta_data['channel_mean'],
        std=meta_data['channel_std']
    )

    train_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'train'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize,
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
        transform=normalize,
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True
    )

    print("Data loading Took {}. # Train {}, # Test {}".format(
        datetime.now() - data_load_start_time,
        len(train_data_loader) * train_batch_size,
        len(val_data_loader) * test_batch_size
    ))

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

            import pdb
            pdb.set_trace()

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

    best_iou = 0

    # Summary file
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    file_handle = open(summary_file, 'w+')
    file_handle.write("Data Set         : {}\n".format(data_set_dir))
    file_handle.write("Train images     : {}\n".format(len(train_set.images)))
    file_handle.write("Val images       : {}\n".format(len(val_set.images)))
    file_handle.write("Train batch size : {}\n".format(train_batch_size))
    file_handle.write("Val batch size   : {}\n".format(test_batch_size))
    file_handle.write("Epochs           : {}\n".format(num_epochs))
    file_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
    file_handle.write("                 : {}\n")
    print(model, file=file_handle)
    file_handle.write("{}\n".format('-' * 80))

    file_handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
    file_handle.write("learning rate    : {}\n".format(learning_rate))
    file_handle.write("Loss             : {}\n".format(criterion.__class__.__name__))
    file_handle.write("{}\n".format('-' * 80))

    file_handle.write("IoU Threshold    : {}\n".format(detect_thres))
    file_handle.write("{}\n".format('-' * 80))

    file_handle.write("Training details\n")
    file_handle.write("Epoch, train_loss, train_iou, val_loss, val_iou, lr\n")

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

        if val_history[epoch][1] > best_iou:
            best_iou = val_history[epoch][1]
            torch.save(
                model.state_dict(),
                os.path.join(results_store_dir, 'best_accuracy.pth')
            )

        file_handle.write("[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {}],\n".format(
            epoch,
            train_history[epoch][0],
            train_history[epoch][1],
            val_history[epoch][0],
            val_history[epoch][1],
            lr_history[epoch]
        ))

    print('Finished Training. Training took {}'.format(datetime.now() - training_start_time))
    file_handle.close()

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
