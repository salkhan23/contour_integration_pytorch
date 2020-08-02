# ---------------------------------------------------------------------------------------
# Generate the Pathfinder on Natural Images Dataset
#
# This is a binary classification task where the model has too determined if the randomly
# selected end points are connected via a smooth edge.
#
# Different from the typical pathfinder task, the inputs are not synthetic but natural
# images. Edge labels are used to select two contours at random. In the connected case,
# the two end points lie on the same contour. While in the not connected case, the
# two end points lie on different contours.
#
# Once the locations of the end points are determined, they are added to the input image.
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
import sys

import torch
from PIL import Image
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader

import dataset_biped
import contour


class OnlineNaturalImagesPathfinder(dataset_biped.BipedDataSet):
    def __init__(self, p_connect=0.5, min_sep_dist=10, end_stop_radius=6, min_contour_len=30,
                 max_contour_len=None, intrpl_ths=None, *args, **kwargs):
        """
        Generate natural pathfinder images/labels on the fly
        Use the BIPED Dataset/loaders

        :param p_connect: probability of class connected.
        :param min_sep_dist: minimum separation between points.
        :param end_stop_radius: Red and Blue concentric circles (Bulls Eye)
            are used as end points
        :param intrpl_ths: BIPED images are resized. Resizing needs to use
             Interpolation. This causes some pixels values to be non binary.
             a Threshold is used to covert back to binary labels. This is
             mostly a robustness strategy. A LIFO list is used to pop out
             different thresholds to try if contours cannot be found.
        """
        if intrpl_ths is None:
            intrpl_ths = [0.1, 0.2, 0.15]
        self.intrpl_ths = intrpl_ths

        self.end_stop_radius = end_stop_radius
        self.p_connect = p_connect
        self.min_sep_dist = min_sep_dist
        self.min_contour_len = min_contour_len
        self.max_contour_len = max_contour_len

        super(OnlineNaturalImagesPathfinder, self).__init__(
            calculate_stats=False, *args, **kwargs)

    def get_img_by_index(self, index, start_point=None, end_point=None):

        img = Image.open(self.images[index]).convert('RGB')
        if self.resize is not None:
            # Resize uses interpolation
            img = self.resize(img)
        img = transform_functional.to_tensor(img)

        if start_point is not None:
            self.add_end_stop(
                img, (start_point[0], start_point[1]), radius=self.end_stop_radius)

        if end_point is not None:
            self.add_end_stop(img, (end_point[0], end_point[1]), radius=self.end_stop_radius)

        return img

    def __getitem__(self, index):
        """
        Override the get item routine
        """
        img = Image.open(self.images[index]).convert('RGB')
        full_label_raw = Image.open(self.labels[index]).convert('L')  # Greyscale
        # print("Index {}".format(index))

        if self.resize is not None:
            # Resize uses interpolation
            img = self.resize(img)
            full_label_raw = self.resize(full_label_raw)

        th_arr = self.intrpl_ths.copy()
        th = th_arr.pop()

        img = transform_functional.to_tensor(img)
        full_label_raw = transform_functional.to_tensor(full_label_raw)

        # Get the binary (thresholded) resized label
        # Necessary for smooth contours after interpolation
        full_label = self._get_threshold_label(full_label_raw, th)

        # [1] Select a random contour
        c1 = []
        dist_start_stop_c1 = 0

        while len(c1) <= 0:
            c1 = contour.get_random_contour(
                full_label[0, ], min_contour_len=self.min_contour_len,
                max_contour_len=self.max_contour_len)

            if len(c1) == 0:
                th_old = th

                if len(th_arr) >= 1:
                    th = th_arr.pop()
                    print("No valid C1 contour found. Change interpolation th {}->{}. "
                          "[Image idx{}: {}]".format(th_old, th, index, self.labels[index]))

                    full_label = self._get_threshold_label(full_label_raw, th)
                else:
                    # Just give up
                    print("No valid C1 contour found for image at index {} and "
                          "interpolation thresholds exhausted".format(index))

                    # Pytorch Data loader does not like None (doesnt know how to add batch dim
                    # Just check output.dim() == 1 (should be 4 in normal case (the img))
                    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            else:
                # Guard against circular contours - we want two distinct end points in
                # each image
                dist_start_stop_c1 = self.get_distance_between_two_points(c1[0], c1[-1])
                while dist_start_stop_c1 <= (self.end_stop_radius * 2) and len(c1) > 1:
                    c1.pop()
                    dist_start_stop_c1 = self.get_distance_between_two_points(c1[0], c1[-1])

                    # # Debug
                    print("Found circle. Distance start stop {:0.2f}. [Image idx {}: {}]".format(
                        dist_start_stop_c1, index, self.labels[index]))
                    # # plt.figure()
                    # # contour.show_contour(full_label[0, ], c1)
                    # circle_found = True

        ideal_c2_dist = dist_start_stop_c1

        # [2] Select the second contour
        # When finding the second contour, make points that are at the desired distance
        # more probable (non uniform probability distribution). Lower values put
        # emphasis on contours at the specified distance. Higher values broaden the
        # search for contours and allow more uniform selection of starting points
        p_scale = 10
        is_overlapping = True
        overlap_count = 0

        c2 = []
        while is_overlapping:
            c2 = []

            while len(c2) <= 0:
                c2 = contour.get_nearby_contour(
                    full_label[0, ], c1[0], ideal_dist=ideal_c2_dist,
                    p_scale=p_scale, max_iterations=200000)

                if len(c2) == 0:
                    th_old = th
                    th = th_arr.pop()
                    print("No valid C2 contour found. Change interpolation th {}->{}.\n "
                          "[Image idx {}: {}]".format(th_old, th, index, self.labels[index]))
                    full_label = self._get_threshold_label(full_label_raw, th)

            # Check that C2 is separate from C1
            for p in c1:
                is_overlapping = self.does_point_overlap_with_contour(p, c2, self.min_sep_dist)
                if is_overlapping:
                    break

            # Check that no points in c2 or near C2 that has a valid contour
            # comes close to C1
            if not is_overlapping:

                p1 = c2[0]
                nearby_points = contour.find_all_edge_points_within_distance(
                    full_label[0, ], p1, self.min_sep_dist)

                nearby_contours = []
                for point in nearby_points:
                    c3 = contour.get_contour_around_point(full_label[0, ], point)

                    if len(c3) is not 0 and c3 not in nearby_contours:
                        nearby_contours.append(c3)

                # Get all points on nearby contours
                points_on_nearby_contours = []
                for c in nearby_contours:
                    points_on_nearby_contours.extend(c)
                # store only unique points
                points_on_nearby_contours = set(points_on_nearby_contours)

                # check that no point is near is within overlapping distance from c1
                for point in points_on_nearby_contours:
                    dist_arr = contour.get_distances_point_to_contour(point, c1)
                    if np.min(dist_arr) < self.end_stop_radius:
                        print("Nearby connected point is too close to C1: Dist={:0.2f} "
                              "[Image idx {}: {}]".format(
                                np.min(dist_arr), index, self.labels[index]))

                        is_overlapping = True

                        # # Debug
                        # temp = np.copy(full_label)
                        # for c in nearby_contours:
                        #     contour.show_contour(temp[0, ], c, value=0.75)
                        # contour.show_contour(temp[0, ], c1, value=0.5)
                        # contour.show_contour(temp[0, ], c2, value=0.25)
                        # plt.scatter(point[1], point[0], color='r', marker='+', s=200)
                        # plt.title("Extension point from C2 too close to C1")
                        # import pdb
                        # pdb.set_trace()

                        break

            if is_overlapping:
                # print("C2 overlaps with C1. count {}".format(overlap_count))
                overlap_count += 1

                if overlap_count >= 100:
                    old_p_scale = p_scale
                    p_scale = p_scale + 10
                    print("C2 overlapped with C1 more than 100 times. Broaden contour search "
                          "space. {}->{}.\n[Image idx {}: {}]".format(
                            old_p_scale, p_scale, index, self.labels[index]))

                    overlap_count = 0

        # [3] Randomly choose to connect the end dot to the contour
        connected = np.random.choice([0, 1], p=[1 - self.p_connect, self.p_connect])
        connected = torch.tensor(connected)

        start_point = None
        end_point = None

        # [4] Locations of End points
        if connected:
            start_point = c1[0]
            end_point = c1[-1]
        else:
            dist_arr = np.array([
                self.get_distance_between_two_points(c1[0], c2[0]),  # c1_start_c2_start
                self.get_distance_between_two_points(c1[0], c2[0]),  # c1_start_c2_stop
                self.get_distance_between_two_points(c1[-1], c2[0]),  # c1_stop_c2_start
                self.get_distance_between_two_points(c1[-1], c2[-1])  # c1_stop_c2_stop
            ])

            # closest in terms of most similar to distance between c1[0] and c1[-1]
            closest_points_idx = (np.abs(dist_arr - ideal_c2_dist)).argmin()

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

        # [5] Create a new label with the two contours - These are only for debug, so
        # highlight the selected contours
        single_contours_label = torch.zeros_like(full_label)
        for p in c1:
            single_contours_label[:, p[0], p[1]] = 0.5  # channel first
            full_label[:, p[0], p[1]] = 0.5
        for p in c2:
            single_contours_label[:, p[0], p[1]] = 0.25  # channel first
            full_label[:, p[0], p[1]] = 0.25

        # Add the starting point
        self.add_end_stop(single_contours_label, start_point, radius=self.end_stop_radius)
        self.add_end_stop(img, start_point, radius=self.end_stop_radius)
        self.add_end_stop(full_label, start_point, radius=self.end_stop_radius)

        # Add the end point
        self.add_end_stop(single_contours_label, end_point, radius=self.end_stop_radius)
        self.add_end_stop(img, end_point, radius=self.end_stop_radius)
        self.add_end_stop(full_label, end_point, radius=self.end_stop_radius)

        dist_between_points = self.get_distance_between_two_points(start_point, end_point)

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
        #     "connected: {}\n Lengths C1={}, C2={}, Dist b/w Endpoints {:0.1f}".format(
        #         connected, len(c1), len(c2),
        #         self.get_distance_between_two_points(start_point, end_point)))
        #
        # import pdb
        # pdb.set_trace()
        # plt.close('all')

        return img, connected, single_contours_label, full_label, dist_between_points, index,\
            torch.tensor(c1), torch.tensor(c2), torch.tensor(start_point), torch.tensor(end_point)

    @staticmethod
    def _get_threshold_label(label, th):
        label_th = torch.zeros_like(label)
        label_th[label >= th] = 1
        label_th[label < th] = 0
        return label_th

    @staticmethod
    def get_distance_between_two_points(p1, p2):
        """
        Get the distance between points p1 & p2. Points specified as  (x,y)
        """
        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return dist

    @staticmethod
    def get_distance_point_and_contour(p, contour1):
        """
        Get distance between a single point (x,y) and a list of points [(x1,y1), (x2,y2) ...]
        """
        contour1 = np.array(contour1)
        dist_arr = np.sqrt((contour1[:, 0] - p[0]) ** 2 + (contour1[:, 1] - p[1]) ** 2)

        return dist_arr

    def does_point_overlap_with_contour(self, p1, contour1, width):
        """
        Is the minimum distance between p1 and any point in contour 1 < width
        p = (x,y)
        contour1 = list of points [(x1,y1), (x2,y2) , ...]
        """
        overlaps = True

        dist_arr = self.get_distance_point_and_contour(p1, contour1)

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

    @staticmethod
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


def create_dataset(data_dir, biped_dataset_type, n_biped_imgs, n_epochs):

    biped_dataset_dir = './data/BIPED/edges'

    # Create the results store directory
    # ----------------------------------
    if os.path.exists(data_dir):
        ans = input("{} already  data. Overwrite ?".format(data_dir))
        if 'y' in ans.lower():
            shutil.rmtree(data_dir)
        else:
            sys.exit()

    imgs_dir = os.path.join(data_dir, 'images')
    indv_contours_labels_dir = os.path.join(data_dir, 'individual_contours_labels')
    full_labels_dir = os.path.join(data_dir, 'full_labels')
    for folder in [imgs_dir, indv_contours_labels_dir, full_labels_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    data_key_filename = os.path.join(data_dir, 'data_key.txt')
    class_labels_filename = os.path.join(data_dir, 'classification_labels.txt')
    map_to_org_imgs_filename = os.path.join(data_dir, 'map_to_original_images.txt')
    distances_filename = os.path.join(data_dir, 'distances_between_points.txt')

    data_key_handle = open(data_key_filename, 'w+')
    class_labels_handle = open(class_labels_filename, 'w+')

    org_imgs_map_handle = open(map_to_org_imgs_filename, 'w+')
    distances_handle = open(distances_filename, 'w+')

    # Setup the Online Pathfinder Data Loader
    # ---------------------------------------
    print("New Dataset will be created @ {}\n{}".format(data_dir, '*' * 80))
    start_time = datetime.now()

    data_set = OnlineNaturalImagesPathfinder(
        data_dir=biped_dataset_dir,
        dataset_type=biped_dataset_type,
        transform=None,
        subset_size=n_biped_imgs,
        resize_size=(256, 256),
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=1,  # must stay as one
        shuffle=False,
        pin_memory=True
    )

    # Main
    # -----
    dist_not_connected = []
    dist_connected = []
    n_imgs_created = 0

    for epoch in range(0, n_epochs):
        print("Processing Epoch {}/{}".format(epoch + 1, n_epochs))

        for iteration, data_loader_out in enumerate(data_loader, 1):

            if data_loader_out[0].dim() == 4:  # if valid image
                imgs, class_labels, indv_contours_label, full_labels, distances, \
                    org_img_idxs, _, _, _, _ = data_loader_out

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
                    fname=os.path.join(
                        indv_contours_labels_dir, 'img_{}.png'.format(n_imgs_created)),
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

    f.savefig(os.path.join(data_dir, 'histogram.jpg'), format='jpg')

    data_key_handle.close()
    class_labels_handle.close()
    org_imgs_map_handle.close()
    distances_handle.close()

    print("Dataset Created. Number images {} from {} unique images. Time {}".format(
        n_imgs_created, n_biped_imgs, datetime.now() - start_time))


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    data_store_dir = './data/pathfinder_natural_images'
    random_seed = 8

    # Total number of images = n_biped_images * n_epochs
    train_n_biped_imgs = 30000
    train_n_epochs = 1

    val_n_biped_imgs = 50
    val_n_epochs = 100

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    print("Creating The Training Dataset ...")
    create_dataset(
        os.path.join(data_store_dir, 'train'),
        'train',
        train_n_biped_imgs,
        train_n_epochs
    )

    print("Creating The Validation Dataset ...")
    create_dataset(
        os.path.join(data_store_dir, 'test'),
        'test',
        val_n_biped_imgs,
        val_n_epochs
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
