# ---------------------------------------------------------------------------------------
# Generate a Test dataset for natural images pathfinder Task
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
import sys

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset_pathfinder_natural_images


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    data_store_dir = './data/pathfinder_natural_images_test'

    # Total images generated will be biped_data_subset_size * n_epochs
    biped_data_subset_size = 500
    n_epochs = 10

    random_seed = 8

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    biped_dataset_dir = './data/BIPED/edges'

    start_time = datetime.now()

    # Create the results store directory
    # ----------------------------------
    if os.path.exists(data_store_dir):
        ans = input("{} already  data. Overwrite ?".format(data_store_dir))
        if 'y' in ans.lower():
            shutil.rmtree(data_store_dir)
        else:
            sys.exit()

    imgs_dir = os.path.join(data_store_dir, 'images')
    indv_contours_labels_dir = os.path.join(data_store_dir, 'individual_contours_labels')
    full_labels_dir = os.path.join(data_store_dir, 'full_labels')
    for folder in [imgs_dir, indv_contours_labels_dir, full_labels_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    data_key_filename = os.path.join(data_store_dir, 'data_key.txt')
    class_labels_filename = os.path.join(data_store_dir, 'classification_labels.txt')
    map_to_org_imgs_filename = os.path.join(data_store_dir, 'map_to_original_images.txt')
    distances_filename = os.path.join(data_store_dir, 'distances_between_points.txt')

    data_key_handle = open(data_key_filename, 'w+')
    class_labels_handle = open(class_labels_filename, 'w+')

    org_imgs_map_handle = open(map_to_org_imgs_filename, 'w+')
    distances_handle = open(distances_filename, 'w+')

    # -----------------------------------------------------------------------------------
    # Setup the Online Pathfinder Data loader
    # -----------------------------------------------------------------------------------
    print("New Dataset will be created @ {}\n{}".format(data_store_dir, '*' * 80))
    start_time = datetime.now()

    data_set = dataset_pathfinder_natural_images.NaturalImagesPathfinder(
        data_dir=biped_dataset_dir,
        dataset_type='train',
        transform=None,
        subset_size=biped_data_subset_size,
        resize_size=(256, 256),
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=1,   # must stay as one
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    dist_not_connected = []
    dist_connected = []
    n_imgs_created = 0

    for epoch in range(0, n_epochs):
        print("Processing Epoch {}/{}".format(epoch + 1, n_epochs))

        for iteration, data_loader_out in enumerate(data_loader, 1):

            imgs, class_labels, indv_contours_label, full_labels, distances, org_img_idxs = \
                data_loader_out

            # Below code assume batch size of 1
            # b_size = imgs.shape[0]

            img = np.transpose(imgs[0, ], axes=(1, 2, 0))
            img_file_name = 'img_{}.png'.format(n_imgs_created)
            plt.imsave(fname=os.path.join(imgs_dir, img_file_name), arr=img.numpy())
            data_key_handle.write("{}\n".format(img_file_name))

            class_label = class_labels[0]
            class_labels_handle.write("{}\n".format(int(class_label)))

            org_img_idx = org_img_idxs[0]
            org_img_name = data_loader.dataset.images[org_img_idx]
            org_imgs_map_handle.write("{}\n".format(org_img_name))

            ind_contour_label = indv_contours_label[0]
            ind_contour_label = np.squeeze(ind_contour_label)
            plt.imsave(
                fname=os.path.join(indv_contours_labels_dir, 'img_{}.png'.format(n_imgs_created)),
                arr=ind_contour_label.numpy())

            full_label = full_labels[0]
            full_label = np.squeeze(full_label)

            plt.imsave(
                fname=os.path.join(full_labels_dir, 'img_{}.png'.format(n_imgs_created)),
                arr=full_label.numpy())

            d_between_points = int(distances[0])
            distances_handle.write("{}\n".format(d_between_points))

            # Store distances between points
            for idx in range(data_loader_out[1].shape[0]):
                if data_loader_out[1][idx]:
                    dist_connected.append(data_loader_out[4][idx])
                else:
                    dist_not_connected.append(data_loader_out[4][idx])

            n_imgs_created += 1

    print("Connected: m={:0.2f}, std={:0.2f}, Not connected: m={:0.2f}, std={:0.2f}".format(
            np.mean(dist_connected), np.std(dist_connected),
            np.mean(dist_not_connected), np.std(dist_not_connected)))

    f, ax_arr = plt.subplots(2, 1, sharex=True, figsize=(9, 9))
    connected_hist = ax_arr[0].hist(dist_connected, bins=np.arange(0, 300, 50))
    ax_arr[0].set_title("Connected. Mean {:0.2f}, Std {:0.2f}".format(
        np.mean(dist_connected), np.std(dist_connected)))

    not_connected_hist = ax_arr[1].hist(dist_not_connected, bins=np.arange(0, 300, 50))
    ax_arr[1].set_title("Not Connected. Mean {:0.2f}, std {:0.2f}".format(
        np.mean(dist_not_connected), np.std(dist_not_connected)))

    f.suptitle("Distribution of distances between end-points. [Counts Connected={}, Not={}]".format(
        len(dist_connected), len(dist_not_connected)))

    y_max = np.max((np.max(connected_hist[0]), np.max(not_connected_hist[0])))
    ax_arr[0].set_ylim(0, y_max * 1.1)
    ax_arr[1].set_ylim(0, y_max * 1.1)

    f.savefig(os.path.join(data_store_dir, 'histogram.jpg'), format='jpg')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    data_key_handle.close()
    class_labels_handle.close()
    org_imgs_map_handle.close()
    distances_handle.close()

    print("Dataset Created. Number images {} from {} unique images. Time {}".format(
        n_imgs_created, biped_data_subset_size, datetime.now() - start_time))

    import pdb
    pdb.set_trace()
