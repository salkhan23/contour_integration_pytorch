# ---------------------------------------------------------------------------------------
# Pytorch Data Set/loader for the pathfinder on natural images task.
#
#
#
# Uses the BIPED dataset and its pytorch data set/loaders
# ---------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset_biped
import contour


def add_end_stop(img1, center=(0, 0), radius=5):
    ax = torch.arange(center[0] - radius, center[0] + radius + 1)
    ay = torch.arange(center[1] - radius, center[1] + radius + 1)

    max_value = torch.max(img1)

    if img1.ndim == 3:
        x_max = img1.shape[1]
        y_max = img1.shape[2]
    else:
        x_max = img1.shape[0]
        y_max = img1.shape[1]

    for x in ax:
        for y in ay:
            # print("processing point ({},{}): Distance from center {}".format(
            #     x, y, np.sqrt(((x - center[0]) ** 2 + (y - center[1]) ** 2))))

            if ((0 <= x < x_max)
                    and (0 <= y < y_max)
                    and ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2):

                x = x.int()
                y = y.int()

                if img1.ndim == 3:
                    img1[:, x, y] = max_value  # Channel First
                else:
                    img1[x, y] = max_value  # Channel First

    return img1


def get_distance_between_two_points(p1, p2):
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)
    return dist


def get_point_within_distance_range(d, p1, img_shape, d_hyst=5):
    """
    Returns the coordinates of a point in the image that lies within
    d +- hysteresis distance away from p1
    """

    x = np.arange(0, img_shape[0])
    y = np.arange(0, img_shape[1])

    xx, yy = np.meshgrid(x, y)
    dist_arr = np.sqrt((xx - p1[0]) ** 2 + (yy - p1[1]) ** 2)

    below_upper_bound = dist_arr < (d + d_hyst)
    above_lower_bound = dist_arr > (d - d_hyst)
    within_range = below_upper_bound * above_lower_bound

    # get all non zero indices
    valid_x, valid_y = np.nonzero(within_range)

    # choose one randomly
    selected_idx = np.random.randint(len(valid_x))

    # finally the end point
    # todo: For some reason these are flipped
    p2 = (valid_y[selected_idx], valid_x[selected_idx])

    # # debug
    # img = np.zeros(img_shape)
    # for x1, y1 in zip(valid_x, valid_y):
    #     img[y1, x1] = 1
    # add_end_stop(torch.from_numpy(img), p1)
    #
    # plt.figure()
    # plt.imshow(img)
    # import pdb
    # pdb.set_trace()

    return p2


def does_point_overlap_with_contour(p1, contour1, width):
    overlaps = True

    contour1 = np.array(contour1)
    dist_arr = np.sqrt((contour1[:, 0] - p1[0]) ** 2 + (contour1[:, 1] - p1[1]) ** 2)

    if not np.any(dist_arr <= width):
        overlaps = False

    # plt.figure()
    # plt.plot(dist_arr)
    # plt.axhline(width, color='red', label='overlapping below')
    # plt.xlabel("Index")
    # plt.legend()
    # plt.title("Is overlapping {}".format(overlaps))

    return overlaps


class NaturalImagesPathfinder(dataset_biped.BipedDataSet):
    """
    returns tuples of img, connected, single_contour_label, full_label
    """

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        full_label = Image.open(self.labels[index]).convert('L')  # Greyscale

        if self.resize is not None:
            img = self.resize(img)
            full_label = self.resize(full_label)  # uses interpolation

        img = transform_functional.to_tensor(img)
        full_label = transform_functional.to_tensor(full_label)
        full_label[full_label >= 0.1] = 1  # necessary for smooth contours after interpolation
        full_label[full_label < 0.1] = 0

        # Select a single contour
        single_contour = contour.get_random_contour(full_label[0, ])
        while len(single_contour) <= 0:
            single_contour = contour.get_random_contour(full_label[0, ])

        dist_contour_start_stop = \
            get_distance_between_two_points(single_contour[0], single_contour[-1])

        # Create a single contour label
        single_contour_label = torch.zeros_like(full_label)
        for point in single_contour:
            single_contour_label[:, point[0], point[1]] = 1  # channel First

        # todo: move to init
        end_stopper_radius = 5
        probability_connect = 0.5
        dist_hyst = 5

        # Add starting dot to image, full label and single_contour label
        start_point = single_contour[0]
        add_end_stop(single_contour_label, start_point, radius=end_stopper_radius)
        add_end_stop(img, start_point, radius=end_stopper_radius)
        add_end_stop(full_label, start_point, radius=end_stopper_radius)

        # Randomly choose to connect the end dot to the contour
        connected = np.random.choice([0, 1], p=[1 - probability_connect, probability_connect])
        connected = connected.astype(bool)

        end_point = None
        if connected:
            # Add the other end stop at the end of the contour
            end_point = single_contour[-1]
        else:
            # Add the other end stop at the same distance from the starting point
            # making sure it does not overlap with any points  on the contour
            is_overlapping = True

            while is_overlapping:

                # Get a point within range
                end_point = get_point_within_distance_range(
                    dist_contour_start_stop,
                    start_point,
                    single_contour_label.shape[1:],  # first is channel
                    dist_hyst,
                )

                is_overlapping = does_point_overlap_with_contour(
                    end_point, single_contour, end_stopper_radius)

        # Add the End point
        if end_point is not None:
            add_end_stop(single_contour_label, end_point, radius=end_stopper_radius)
            add_end_stop(img, end_point, radius=end_stopper_radius)
            add_end_stop(full_label, end_point, radius=end_stopper_radius)

        # # Debug
        # # ------
        # f, ax_arr = plt.subplots(1, 3, figsize=(15, 6))
        #
        # display_img = (img - img.min()) / (img.max() - img.min())
        # display_img = display_img.numpy()
        # display_img = np.transpose(display_img, axes=(1, 2, 0))
        # ax_arr[0].imshow(display_img)
        #
        # single_contour_label = single_contour_label.numpy()
        # single_contour_label = np.squeeze(single_contour_label)
        # ax_arr[1].imshow(single_contour_label)
        # ax_arr[1].set_title("Single Contour Label")
        #
        # full_label = full_label.numpy()
        # full_label = np.squeeze(full_label)
        # ax_arr[2].imshow(full_label)
        # ax_arr[2].set_title("Full label")
        #
        # f.suptitle(
        #     "CLen {}.\nEnd Points Connected {},\ndistance contour start stop={:0.2f}, "
        #     "\ndistance start stop={:0.2f}".format(
        #         len(single_contour), connected, dist_contour_start_stop,
        #         get_distance_between_two_points(start_point, end_point)
        #     ))
        #
        # import pdb
        # pdb.set_trace()
        # plt.close(f)

        return img, connected, single_contour_label, full_label, index


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    base_dir = './data/BIPED/edges'

    random_seed = 7

    train_batch_size = 32
    test_batch_size = 1

    # Immutable ----------------------
    import matplotlib.pyplot as plt

    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Imagenet normalization
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]
    pre_process_transforms = transforms.Normalize(mean=ch_mean, std=ch_std)

    # Training Loader
    # -----------------------------------------------------------------------------------
    print("Setting up the Train Data Loaders {}".format('*' * 30))
    start_time = datetime.now()

    train_set = NaturalImagesPathfinder(
        data_dir=base_dir,
        dataset_type='train',
        transform=pre_process_transforms,
        subset_size=100,
        resize_size=(256, 256)
    )

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Setting up the train data loader took {}".format(datetime.now() - start_time))

    train_generator = training_data_loader.__iter__()
    imgs, class_labels, single_contour_labels, full_labels, org_img_idx_arr = \
        train_generator.__next__()

    for img_idx in range(imgs.shape[0]):
        image = imgs[img_idx, ].numpy()
        image = np.transpose(image, axes=(1, 2, 0))

        class_label = class_labels[img_idx]

        s_label = single_contour_labels[img_idx]
        s_label = np.squeeze(s_label)

        f_label = full_labels[img_idx]
        f_label = np.squeeze(f_label)

        fig, axis_arr = plt.subplots(1, 3)
        d_img = (image - image.min()) / (image.max() - image.min())
        axis_arr[0].imshow(d_img)
        axis_arr[0].set_title("Input Image")

        axis_arr[1].imshow(s_label)
        axis_arr[1].set_title("Debug Single Contour Label")

        axis_arr[2].imshow(f_label)
        axis_arr[2].set_title("Debug Full Contour Label")

        fig.suptitle("Classification Label {}. img name {}".format(
            class_label, training_data_loader.dataset.images[org_img_idx_arr[img_idx]]))

        import pdb
        pdb.set_trace()

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
