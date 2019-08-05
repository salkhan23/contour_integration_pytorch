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
import torch.nn.functional as nn_functional
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset
import fields1993_stimuli


class ClassifierHead(nn.Module):
    def __init__(self, num_channels):
        super(ClassifierHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels // 2)

        self.conv2 = nn.Conv2d(
            in_channels=num_channels // 2,
            out_channels=num_channels // 4,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels // 4)

        self.conv3 = nn.Conv2d(
            in_channels=num_channels // 4,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False
        )

    def forward(self, x):
        x = nn_functional.relu(self.bn1(self.conv1(x)))
        x = nn_functional.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x


class CurrentSubtractiveInhibition(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=3):

        super(CurrentSubtractiveInhibition, self).__init__()

        # Parameters
        self.n_iters = n_iters  # Number of recurrent steps
        self.a = 0.5  # Weighting factor for combining excitatory recurrence and feed-forward. Should be [Nx1]
        self.b = 0.5  # Weighting factor for combining inhibitory recurrence and feed-forward  Should be [Nx1]
        # N = number of feature channels
        # TODO: 1. find proper setting, 2. can these be learned?

        self.edge_out_ch = edge_out_ch
        self.a = nn.Parameter(torch.rand(edge_out_ch))
        self.b = nn.Parameter(torch.rand(edge_out_ch))

        self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        self.j_xy = nn.Parameter(torch.rand(edge_out_ch))
        self.j_yx = nn.Parameter(torch.rand(edge_out_ch))

        self.e_bias = nn.Parameter(torch.rand(edge_out_ch))
        self.i_bias = nn.Parameter(torch.rand(edge_out_ch))

        # Layers
        # ------
        # First layer is AlexNet Edge Extraction (with out bias)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=edge_out_ch,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=False
        )
        # use pre-trained Alexnet weights
        alexnet_edge_detect_kernels = torchvision.models.alexnet(pretrained=True).features[0]
        self.conv1.weight.data.copy_(alexnet_edge_detect_kernels.weight.data)

        # Additional Batch normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=64)

        # Recurrent lateral connection parts
        # TODO: What is the spatial extent of the kernel for one iteration
        # TODO: Adjust padding to match dimensions
        self.lateral_e = nn.Conv2d(
            in_channels=edge_out_ch, out_channels=edge_out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.lateral_i = nn.Conv2d(
            in_channels=edge_out_ch, out_channels=edge_out_ch, kernel_size=7, stride=1, padding=3, bias=False)

        self.post = ClassifierHead(edge_out_ch)

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # -------------------------------------------------------------------------------
        # Contour Integration
        # -------------------------------------------------------------------------------
        x = torch.zeros_like(ff)    # state of excitatory neurons
        f_x = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons
        y = torch.zeros_like(ff)    # state of inhibitory neurons
        f_y = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons

        for i in range(self.n_iters):

            # print("processing iteration {}".format(i))

            # crazy broadcasting. dim=1 tell torch that this dim needs to be broadcast
            x = (1 - self.a.view(1, 64, 1, 1)) * x + \
                self.a.view(1, 64, 1, 1) * (
                    (self.j_xx.view(1, 64, 1, 1) * f_x) -
                    (self.j_xy.view(1, 64, 1, 1) * f_y) +
                    ff +
                    self.e_bias.view(1, 64, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_e(f_x))
                )
            # TODO: first f_x should be one dimensional for eah channel, second one should include neighbors

            f_x = nn.functional.relu(x)

            y = (1 - self.b.view(1, 64, 1, 1) * y) + \
                self.b.view(1, 64, 1, 1) * (
                    (self.j_yx.view(1, 64, 1, 1) * f_x) +
                    self.i_bias.view(1, 64, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_i(f_x))
                )

            f_y = nn.functional.relu(y)

        # Post processing
        out = self.post(f_x)

        return out


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()

    train_batch_size = 16
    test_batch_size = 1
    device = torch.device("cuda")
    learning_rate = 0.001
    num_epochs = 20

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("Loading Model")
    model = CurrentSubtractiveInhibition().to(device)
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

        print("Epoch {} Finished, Loss = {}".format(epoch, epoch_loss / len(train_data_loader)))

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
