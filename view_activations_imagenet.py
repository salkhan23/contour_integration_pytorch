import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import torch.nn as nn

import train_imagenet
import models.piech_models as piech_models
import models.control_models as control_models

edge_extract_act = []
contour_integration_act = []


def edge_extraction_hook(self, input_l, output):
    edge_extract_act.append(output.cpu().detach().numpy())


def contour_integration_hook(self, input_l, output):
    contour_integration_act.append(output.cpu().detach().numpy())


if __name__ == "__main__":

    random_seed = 7

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

    batch_size = 1
    workers = 1
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4

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

    #  Criterion
    # -----------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # # Validate
    # # ----------------------------------------------------------------------------------

    # val_loss, val_acc1, val_acc5 = train_imagenet.validate(val_loader, net, criterion, args)
    # print("Loss: {}, val_acc1 {}, val_acc5 {}".format(val_loss, val_acc1, val_acc5))

    # Add Activation Hooks
    # ----------------------------------------------------------------------------------
    # Register the hooks
    net.conv1.conv1.register_forward_hook(edge_extraction_hook)
    net.conv1.register_forward_hook(contour_integration_hook)

    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        out = net(images)

        # Edge Extraction Output
        y = np.squeeze(edge_extract_act[i])
        y1 = np.sum(y, axis=0)  # Sum activations across all channels

        # Contour Extraction Out
        z = np.squeeze(contour_integration_act[i])
        z1 = np.sum(z, axis=0)

        # Input image
        image = images.cpu().detach().numpy()
        image = np.squeeze(image)
        image = np.transpose(image, axes=(1, 2, 0))
        plt.figure()
        plt.imshow(image)
        plt.title("Input (normalized) Image")

        # Display Figures
        f, ax_arr = plt.subplots(1, 2, sharey=True, squeeze=True, figsize=(12, 6))
        p1 = ax_arr[0].imshow(y1, cmap='seismic', vmax=np.mean(y1) + np.std(y1))
        ax_arr[0].set_title("Edge Extract Out (Mean={:0.4f})".format(y1.mean()))
        plt.colorbar(p1, ax=ax_arr[0], orientation='horizontal')

        p2 = ax_arr[1].imshow(z1, cmap='seismic', vmax=np.mean(z1) + np.std(z1))
        ax_arr[1].set_title("Contour Integration Out (Mean={:0.4f})".format(z1.mean()))
        plt.colorbar(p2, ax=ax_arr[1], orientation='horizontal')

        import pdb
        pdb.set_trace()
        plt.close('all')
