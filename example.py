# ---------------------------------------------------------------------------------------
# Generate a Test dataset for natural images pathfinder Task
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset_pathfinder_natural_images


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    data_store_dir = './data/pathfinder_natural_images/test'

    random_seed = 5
    n_epochs = 10

    contour_lengths_bins = [20, 50, 100, 150, 200]

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    biped_dataset_dir = './data/BIPED/edges'

    # -----------------------------------------------------------------------------------
    # Setup the online Pathfinder Data loader
    # -----------------------------------------------------------------------------------
    # Imagenet normalization
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]
    pre_process_transforms = transforms.Normalize(mean=ch_mean, std=ch_std)

    print("Setting up the Data Loaders {}".format('*' * 30))
    start_time = datetime.now()

    data_set = dataset_pathfinder_natural_images.NaturalImagesPathfinder(
        data_dir=biped_dataset_dir,
        dataset_type='train',
        transform=pre_process_transforms,
        subset_size=1000,
        resize_size=(256, 256),
        # min_contour_len=50,
        # max_contour_len=50
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    print("Setting up the train data loader took {}".format(datetime.now() - start_time))

    dist_not_connected = []
    dist_connected = []

    for epoch in range(0, n_epochs):
        for iteration, data_loader_out in enumerate(data_loader, 1):
            # print("Iteration {}".format(iteration))

            # imgs = data_loader_out[0]
            # class_labels = data_loader_out[1]
            # single_contours_labels = data_loader_out[2]
            # full_labels = data_loader_out[3]
            #
            # idx = 0
            #
            # img = imgs[idx, ]
            # class_label = class_labels[idx, ]
            # single_contour_label = single_contours_labels[idx, ]
            # full_label = full_labels[idx, ]
            #
            # img = np.transpose(img, axes=(1, 2, 0))
            # display_img = (img - img.min()) / (img.max() - img.min())
            #
            # f, ax_arr = plt.subplots(1, 3, figsize=(15,5))
            # ax_arr[0].imshow(display_img)
            # ax_arr[1].imshow(np.squeeze(single_contour_label))
            # ax_arr[1].set_title("Connected {}".format(class_label))
            # ax_arr[2].imshow(np.squeeze(full_label))
            #
            # import pdb
            # pdb.set_trace()

            for idx in range(data_loader_out[1].shape[0]):
                if data_loader_out[1][idx]:
                    dist_connected.append(data_loader_out[4][idx])
                else:
                    dist_not_connected.append(data_loader_out[4][idx])

        print("Connected: m={:0.2f}, std={:0.2f}, Not connected: m={:0.2f}, std={:0.2f}".format(
                np.mean(dist_connected), np.std(dist_connected),
                np.mean(dist_not_connected), np.std(dist_not_connected)))

    f, ax_arr = plt.subplots(2, 1, sharex=True)
    ax_arr[0].hist(dist_connected)
    ax_arr[0].set_title("Connected. Mean {:0.2f}, Std {:0.2f}".format(
        np.mean(dist_connected), np.std(dist_connected)))

    ax_arr[1].hist(dist_not_connected)
    ax_arr[1].set_title("Not Connected. Mean {:0.2f}, std {:0.2f}".format(
        np.mean(dist_not_connected), np.std(dist_not_connected)))

    f.suptitle("Distribution of distances between end-points. [Counts Connected={}, Not={}]".format(
        len(dist_connected), len(dist_not_connected)))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
