import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn as nn

import train_imagenet
import models.piech_models as piech_models
import models.control_models as control_models

contour_int_in_act = []
contour_int_out_act = []


def contour_int_in(self, input_l, output):
    contour_int_in_act.append(nn.functional.relu(output).cpu().detach().numpy())


def contour_integration_hook(self, input_l, output):
    contour_int_out_act.append(output.cpu().detach().numpy())


if __name__ == "__main__":

    random_seed = 7

    batch_size = 1
    workers = 1

    # Contour Integration Model
    model_to_embed = piech_models.CurrentSubtractiveInhibition(use_class_head=False)
    net = train_imagenet.embed_resnet50(model_to_embed)
    saved_model = \
        './results/imagenet_classification/' \
        'Resnet50_20190907_162401_pretrained_with_contour_integration/best_accuracy.pth'

    # # Control Model
    # model_to_embed = control_models.CmMatchParameters(use_class_head=False)
    # net = train_imagenet.embed_resnet50(model_to_embed)
    # saved_model = \
    #     './results/imagenet_classification/' \
    #     'Resnet50_20190911_175651_pretrained_with_control_layer/best_accuracy.pth'

    # --------------------------
    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train_imagenet stores the not the whole state of everything. Not just the weights.
    # this is similar to how resume option is used in the train imagenet script
    print("Loading model weights")
    checkpoint = torch.load(saved_model)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)

    mpl.rcParams.update({
        'font.size': 18,
        'lines.linewidth': 3}
    )

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class Args:
        def __init__(self):
            self.print_freq = 5000
            self.gpu = device

    args = Args()

    # Data Loaders
    # -----------------------------------------------------------------------------------
    print("Setting Up the Data Loaders ...")
    image_net_dir = '/home/salman/workspace/keras/my_projects/contour_integration/data/imagenet-data'

    valdir = os.path.join(image_net_dir, 'val')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    # #  Validate
    # # -----------------------------------------------------------------------------------
    # lr = 0.1
    # momentum = 0.9
    # weight_decay = 1e-4
    #
    # criterion = nn.CrossEntropyLoss().cuda(device)
    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     lr,
    #     momentum=momentum,
    #     weight_decay=weight_decay
    # )
    #
    # val_loss, val_acc1, val_acc5 = train_imagenet.validate(val_loader, net, criterion, args)
    # print("Loss: {}, val_acc1 {}, val_acc5 {}".format(val_loss, val_acc1, val_acc5))

    # Add Activation Hooks
    # ----------------------------------------------------------------------------------
    # Register the hooks
    net.conv1.bn1.register_forward_hook(contour_int_in)
    net.conv1.register_forward_hook(contour_integration_hook)

    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        out = net(images)

        # Edge Extraction Output
        y = np.squeeze(contour_int_in_act[i])
        y1 = np.sum(y, axis=0)  # Sum activations across all channels

        # Contour Extraction Out
        z = np.squeeze(contour_int_out_act[i])
        z1 = np.sum(z, axis=0)

        # Input image
        image = images.cpu().detach().numpy()
        image = np.squeeze(image)
        image = np.transpose(image, axes=(1, 2, 0))

        f, ax_arr = plt.subplots(1, 3, sharey=True, squeeze=True, figsize=(18, 6))

        ax_arr[0].imshow(image)
        ax_arr[0].set_title('Input Image')

        p1 = ax_arr[1].imshow(y1, cmap='seismic', vmax=np.mean(y1) + np.std(y1))
        ax_arr[1].set_title("Edge Extract Out (Mean={:0.2f})".format(y1.mean()))
        plt.colorbar(p1, ax=ax_arr[1], orientation='horizontal')

        p2 = ax_arr[2].imshow(z1, cmap='seismic', vmax=np.mean(z1) + np.std(z1))
        ax_arr[2].set_title("Cont Int Out (Mean={:0.2f})".format(z1.mean()))
        plt.colorbar(p2, ax=ax_arr[2], orientation='horizontal')

        # Note: Not clear how to look at contour gain across 64 dimensions, try different approaches

        # 1. Based on summed responses
        gain1 = z1 / (y1 + 1e-5)
        diff1 = z1 - y1

        f, ax_arr = plt.subplots(1, 2, sharey=True, squeeze=True, figsize=(12, 6))
        f.suptitle("Activations summed across all 64 channels")

        p0 = ax_arr[0].imshow(gain1, cmap='seismic')
        plt.colorbar(p0, ax=ax_arr[0], orientation='horizontal')
        ax_arr[0].set_title('Gain')

        p1 = ax_arr[1].imshow(diff1, cmap='seismic')
        plt.colorbar(p1, ax=ax_arr[1], orientation='horizontal')
        ax_arr[1].set_title('Diff')

        # 2. Max active neurons across each channel
        y2 = np.max(y, axis=0)
        z2 = np.max(z, axis=0)
        gain2 = z2 / (y2 + 1e-5)
        diff2 = z2 - y2

        f, ax_arr = plt.subplots(1, 2, sharey=True, squeeze=True, figsize=(12, 6))
        f.suptitle("Max active neuron across all 64 channels")

        p0 = ax_arr[0].imshow(gain2, cmap='seismic')
        plt.colorbar(p0, ax=ax_arr[0], orientation='horizontal')
        ax_arr[0].set_title('Gain')

        p1 = ax_arr[1].imshow(diff2, cmap='seismic')
        plt.colorbar(p1, ax=ax_arr[1], orientation='horizontal')
        ax_arr[1].set_title('Diff')

        # # 3. Individual Channels
        # for ch_idx in range(y.shape[0]):
        #     y3 = y[ch_idx, ]
        #     z3 = z[ch_idx, ]
        #     gain3 = z3 / (y3 + 1e-5)
        #     diff3 = z3 - y3
        #
        #     f.suptitle("Contour Gain @ channel {}".format(ch_idx))
        #
        #     f, ax_arr = plt.subplots(1, 2, sharey=True, squeeze=True, figsize=(12, 6))
        #
        #     p0 = ax_arr[0].imshow(gain3, cmap='seismic')
        #     plt.colorbar(p0, ax=ax_arr[0], orientation='horizontal')
        #     ax_arr[0].set_title('Gain')
        #
        #     p1 = ax_arr[1].imshow(diff3, cmap='seismic')
        #     plt.colorbar(p1, ax=ax_arr[1], orientation='horizontal')
        #     ax_arr[1].set_title('Diff')
        #
        #     import pdb
        #     pdb.set_trace()
        #     plt.close('all')

        import pdb
        pdb.set_trace()
        plt.close('all')
