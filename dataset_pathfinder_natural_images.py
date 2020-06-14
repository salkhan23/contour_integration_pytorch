# ---------------------------------------------------------------------------------------
# Pytorch Data Set/loader for the pathfinder on natural images task.
#
# Uses the BIPED dataset and its pytorch data set/loaders
# ---------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset_biped
import contour


def get_distance_between_two_points(p1, p2):
    """
    Get the distance between points p1 & p2. Points specified as  (x,y)
    """
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)
    return dist


def get_distance_point_and_contour(p, contour1):
    """
    Get distance between a single point (x,y) and a list of points [(x1,y1), (x2,y2) ...]
    """
    contour1 = np.array(contour1)
    dist_arr = np.sqrt((contour1[:, 0] - p[0]) ** 2 + (contour1[:, 1] - p[1]) ** 2)

    return dist_arr


def does_point_overlap_with_contour(p1, contour1, width):
    """
    Is the minimum distance between p1 and any point in contour 1 < width
    p = (x,y)
    contour1 = list of points [(x1,y1), (x2,y2) , ...]
    """
    overlaps = True

    dist_arr = get_distance_point_and_contour(p1, contour1)

    if not np.any(dist_arr <= width):
        overlaps = False

    # # Debug
    # plt.figure()
    # plt.plot(dist_arr)
    # plt.axhline(width, color='red', label='overlapping below')
    # plt.xlabel("Index")
    # plt.legend()
    # plt.title("Is overlapping {}".format(overlaps))

    return overlaps


def add_end_stop(img1, center=(0, 0), radius=8):
    """
    img1 should be 3d (channel first)
    """
    ax = torch.arange(center[0] - radius, center[0] + radius + 1)
    ay = torch.arange(center[1] - radius, center[1] + radius + 1)

    max_value = torch.max(img1)
    n_channels = img1.shape[0]

    # Labels
    if n_channels == 1:
        for x in ax:
            for y in ay:
                if ((0 <= x < img1.shape[1]) and (0 <= y < img1.shape[2])
                        and ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2):
                    x = x.int()
                    y = y.int()
                    img1[:, x, y] = 0
                    img1[0, x, y] = max_value

    else:  # images
        for x in ax:
            for y in ay:
                if (0 <= x < img1.shape[1]) and (0 <= y < img1.shape[2]):

                    d = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    x = x.int()
                    y = y.int()

                    if d < 2:
                        img1[:, x, y] = 0
                        img1[0, x, y] = max_value
                    elif d < 4:
                        img1[:, x, y] = 0
                        img1[2, x, y] = max_value
                    elif d <= radius:
                        img1[:, x, y] = 0
                        img1[0, x, y] = max_value
    return img1


class NaturalImagesPathfinder(dataset_biped.BipedDataSet):
    """
    TODO:
    """
    def __init__(self, p_connect=0.5, min_sep_dist=20, end_stop_radius=8, *args, **kwargs):

        self.end_stop_radius = end_stop_radius
        self.p_connect = p_connect
        self.min_sep_dist = min_sep_dist

        super(NaturalImagesPathfinder, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        full_label = Image.open(self.labels[index]).convert('L')  # Greyscale
        print("Index {}".format(index))

        if self.resize is not None:
            img = self.resize(img)
            full_label = self.resize(full_label)  # uses interpolation

        img = transform_functional.to_tensor(img)
        full_label = transform_functional.to_tensor(full_label)
        full_label[full_label >= 0.15] = 1  # necessary for smooth contours after interpolation
        full_label[full_label < 0.15] = 0

        # [1] Select a random contour
        c1 = []
        while len(c1) <= 0:
            c1 = contour.get_random_contour(full_label[0, ])

        dist_start_stop_c1 = get_distance_between_two_points(c1[0], c1[-1])
        # Guard against circular contours
        ideal_dist = np.max((dist_start_stop_c1, self.min_sep_dist))

        # [2] Select a second non-overlapping contour
        is_overlapping = True
        overlap_count = 0
        # scale the probabilities for the second contour. Lower values put emphasis for
        # contours at the specified distance. Higher values broaden the search for contours
        # more uniform selection of contours.
        p_scale = 10

        c2 = []
        while is_overlapping:
            c2 = []
            while len(c2) <= 0:
                c2 = contour.get_nearby_contour(
                    full_label[0, ], c1[0], ideal_dist=ideal_dist, p_scale=p_scale)

            for p in c1:
                is_overlapping = does_point_overlap_with_contour(p, c2, self.min_sep_dist)
                if is_overlapping:
                    # print("C2 overlaps with C1. count {}".format(overlap_count))
                    overlap_count += 1

                    if overlap_count >= 100:
                        old_p_scale = p_scale
                        p_scale = p_scale + 10
                        print("C2 overlapped with C1 more than 100 times. "
                              "Broaden contour search space. {}->{}".format(old_p_scale, p_scale))

                        overlap_count = 0

                    break

        # Randomly choose to connect the end dot to the contour
        connected = np.random.choice([0, 1], p=[1 - self.p_connect, self.p_connect])
        connected = connected.astype(bool)

        start_point = None
        end_point = None

        if connected:
            start_point = c1[0]
            end_point = c1[-1]
        else:
            dist_arr = np.array([
                get_distance_between_two_points(c1[0], c2[0]),      # c1_start_c2_start
                get_distance_between_two_points(c1[0], c2[0]),      # c1_start_c2_stop
                get_distance_between_two_points(c1[-1], c2[0]),     # c1_stop_c2_start
                get_distance_between_two_points(c1[-1], c2[-1])     # c1_stop_c2_stop
            ])

            # closest in terms of most similar to distance between c1[0] and c1[-1]
            closest_points_idx = (np.abs(dist_arr - ideal_dist)).argmin()

            if closest_points_idx == 0:
                start_point = c1[0]
                end_point = c2[0]
            elif closest_points_idx == 1:
                start_point = c1[0]
                end_point = c2[-1]
            elif closest_points_idx == 2:
                start_point = c1[-1]
                end_point = c2[0]
            elif closest_points_idx == 3:
                start_point = c1[-1]
                end_point = c2[-1]

        # [3] Create a new label with the two contours
        single_contours_label = torch.zeros_like(full_label)
        for p in c1:
            single_contours_label[:, p[0], p[1]] = 1  # channel first
        for p in c2:
            single_contours_label[:, p[0], p[1]] = 0.5  # channel first

        # Add the starting point
        add_end_stop(single_contours_label, start_point, radius=self.end_stop_radius)
        add_end_stop(img, start_point, radius=self.end_stop_radius)
        add_end_stop(full_label, start_point, radius=self.end_stop_radius)

        # Add the end point
        add_end_stop(single_contours_label, end_point, radius=self.end_stop_radius)
        add_end_stop(img, end_point, radius=self.end_stop_radius)
        add_end_stop(full_label, end_point, radius=self.end_stop_radius)

        dist_between_points = get_distance_between_two_points(start_point, end_point)

        # # Debug
        # # ------
        # f, ax_arr = plt.subplots(1, 3, figsize=(15, 6))
        #
        # display_img = (img - img.min()) / (img.max() - img.min())
        # display_img = display_img.numpy()
        # display_img = np.transpose(display_img, axes=(1, 2, 0))
        # ax_arr[0].imshow(display_img)
        #
        # single_contours_label = single_contours_label.numpy()
        # single_contours_label = np.squeeze(single_contours_label)
        # ax_arr[1].imshow(single_contours_label)
        # ax_arr[1].set_title("Single Contour Label: {}".format(connected))
        #
        # full_label = full_label.numpy()
        # full_label = np.squeeze(full_label)
        # ax_arr[2].imshow(full_label)
        # ax_arr[2].set_title("Full label")
        #
        # f.suptitle(
        #     "Are connected {}, [Lengths C1={},C2={}, Dist between End-points {:0.1f}, "
        #     "Distance between contour start/stop C1={:0.1f}, C2={:0.1f}]".format(
        #         connected, len(c1), len(c2),
        #         get_distance_between_two_points(start_point, end_point),
        #         get_distance_between_two_points(c1[0], c1[-1]),
        #         get_distance_between_two_points(c2[0], c2[-1])))
        #
        # results_dir = './results/sample_images'
        # if not os.path.exists(results_dir):
        #     os.makedirs(results_dir)
        #
        # f.savefig(
        #     os.path.join(
        #         results_dir, "img{}_{}".format(index, self.images[index].split('/')[-1])),
        #     format='jpg'
        # )
        #
        # import pdb
        # pdb.set_trace()
        #
        # plt.close(f)

        return img, connected, single_contours_label, full_label, dist_between_points, index


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    base_dir = './data/BIPED/edges'

    random_seed = 5

    train_batch_size = 32
    test_batch_size = 1

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    # Imagenet normalization
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]
    pre_process_transforms = transforms.Normalize(mean=ch_mean, std=ch_std)

    print("Setting up the Data Loaders {}".format('*' * 30))
    start_time = datetime.now()

    data_set = NaturalImagesPathfinder(
        data_dir=base_dir,
        dataset_type='train',
        transform=pre_process_transforms,
        subset_size=500,
        resize_size=(256, 256)
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Setting up the train data loader took {}".format(datetime.now() - start_time))

    # -----------------------------------------------------------------------------------
    # Get the distribution of distances between points
    # -----------------------------------------------------------------------------------
    dist_not_connected = []
    dist_connected = []

    for iteration, data_loader_out in enumerate(data_loader, 1):
        # print("Iteration {}".format(iteration))

        for idx in range(data_loader_out[1].shape[0]):
            if data_loader_out[1][idx]:
                dist_connected.append(data_loader_out[4][idx])
            else:
                dist_not_connected.append(data_loader_out[4][idx])

    print("Connected: m={:0.2f}, std={:0.2f}, Not connected: m={:0.2f}, std={:0.2f}".format(
        np.mean(dist_connected), np.std(dist_connected),
        np.mean(dist_not_connected), np.std(dist_not_connected)))

    # Histogram of distances between points
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
