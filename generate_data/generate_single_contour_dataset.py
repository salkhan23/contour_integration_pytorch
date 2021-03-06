# ---------------------------------------------------------------------------------------
# Generate single contour dataset.
# Given  an input dataset of images and labels, this scripts selects contours of particular
# lengths and creates a new dataset of single contours labels and images.
#
# This dataset is not for training and only contains validation images
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from PIL import Image
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import contour

MAX_DUPLICATE_COUNT = 500

if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7

    data_set_type = 'train_no_aug'

    data_store_dir = './data/single_contour_natural_images_test'
    print("Dataset will be stored @ {}".format(data_store_dir))

    contour_lengths_bins = [20, 50, 100, 150, 200]

    script_start_time = datetime.now()

    min_pixels_per_bin = 50000

    # Immutable
    # --------------------------------------
    plt.ion()
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # List of Images/Labels
    # -----------------------------------------------------------------------------------
    data_dir = './data/BIPED/edges'
    data_key_file = os.path.join(data_dir, '{}_rgb.lst'.format(data_set_type))

    valid_data_set_type = ['test', 'train', 'train_no_aug']
    data_set_type = data_set_type.lower()

    if data_set_type not in valid_data_set_type:
        raise Exception("Invalid Dataset type {}. Must be one of {}".format(
            data_set_type, valid_data_set_type))

    with open(data_key_file, 'r+') as f_handle:
        data_key = f_handle.readlines()
    data_key = [line.split(' ')[0] for line in data_key]

    if data_set_type == 'train_no_aug':
        img_dir = os.path.join(data_dir, 'imgs', 'train')
        label_dir = os.path.join(data_dir, 'edge_maps', 'train')
    else:
        img_dir = os.path.join(data_dir, 'imgs', data_set_type)
        label_dir = os.path.join(data_dir, 'edge_maps', data_set_type)

    list_of_imgs = [os.path.join(img_dir, file) for file in data_key]
    list_of_labels = [os.path.join(label_dir, file.split('.')[0] + '.png') for file in data_key]
    print("Number of images in data set {}".format(len(data_key)))

    if len(list_of_imgs) != len(list_of_labels):
        raise Exception("Number of Source images {} and labels {} do not match".format(
            len(list_of_imgs), list_of_labels))

    img_names = [img.split('.')[0] for img in list_of_imgs]
    label_names = [label.split('.')[0] for label in list_of_labels]

    if img_names != label_names:
        raise Exception("List of images and labels are not equal")

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    images_per_bin_arr = np.zeros_like(contour_lengths_bins)
    pixels_per_bin_arr = np.zeros_like(contour_lengths_bins)

    for bin_idx, bin_len in enumerate(contour_lengths_bins):

        min_len = bin_len
        max_len = 500
        bin_start_time = datetime.now()

        if bin_idx < len(contour_lengths_bins) - 1:
            max_len = contour_lengths_bins[bin_idx + 1]

        print("Generating Images/labels with contour of length in [{}, {}]".format(
            min_len, max_len - 1))

        # Create the bin data store directories
        bin_imgs_dir = \
            os.path.join(data_store_dir, 'images', 'len_{}_{}'.format(bin_len, max_len - 1))
        bin_labels_dir = \
            os.path.join(data_store_dir, 'labels', 'len_{}_{}'.format(bin_len, max_len - 1))
        bin_full_labels_dir = \
            os.path.join(data_store_dir, 'labels_full', 'len_{}_{}'.format(bin_len, max_len - 1))

        if not os.path.exists(bin_imgs_dir):
            os.makedirs(bin_imgs_dir)
        if not os.path.exists(bin_labels_dir):
            os.makedirs(bin_labels_dir)
        if not os.path.exists(bin_full_labels_dir):
            os.makedirs(bin_full_labels_dir)

        # Randomly choose an image from the dataset to store the labels
        data_idx = np.random.randint(0, len(list_of_imgs))

        bin_pixel_count = 0
        bin_img_count = 0

        list_of_contours = []  # list of saved (image name, contour) to prevent duplicating contours
        duplicates_count = 0

        while (bin_pixel_count <= min_pixels_per_bin) and (duplicates_count < MAX_DUPLICATE_COUNT):

            # Randomly choose an image from the start database to search for contours
            data_idx = np.random.randint(0, len(list_of_imgs))

            img_file = list_of_imgs[data_idx]
            label_file = list_of_labels[data_idx]

            img = Image.open(img_file).convert('RGB')
            img = img.resize((256, 256))
            img = np.array(img)

            label = Image.open(label_file).convert('L')  # [0, 1] Mask
            label = label.resize((256, 256), Image.BILINEAR)  # Need for continuous edges
            label = np.array(label) / 255.0
            label[label >= 0.1] = 1  # needed because of interpolation
            label[label < 0.1] = 0

            # Find a single contour with length in the bin
            single_contour = contour.get_random_contour(
                label, min_contour_len=min_len, max_contour_len=max_len)
            len_single_contour = len(single_contour)

            # check for uniqueness
            if len_single_contour > 0:
                is_unique = True
                for (stored_img_name, stored_contour) in list_of_contours:
                    if stored_img_name == img_file:
                        if set(single_contour) == set(stored_contour):
                            duplicates_count += 1
                            print("Duplicate contour {}".format(duplicates_count))
                            is_unique = False

                if is_unique:
                    print("Len {} contour found. n_pixels {}, n_images {}. ".format(
                        len_single_contour, bin_pixel_count, bin_img_count))

                    list_of_contours.append((img_file, single_contour))

                    # Create Single contour label
                    single_contour_label = np.zeros_like(label)
                    for point in single_contour:
                        single_contour_label[point[0], point[1]] = 1

                    # save the image and label
                    img_name = 'img_{}_clen_{}.jpg'.format(bin_img_count, len_single_contour)
                    label_name = 'img_{}_clen_{}.png'.format(bin_img_count, len_single_contour)

                    plt.imsave(fname=os.path.join(bin_imgs_dir, img_name), arr=img)
                    plt.imsave(fname=os.path.join(bin_labels_dir, label_name),
                               arr=single_contour_label)
                    plt.imsave(fname=os.path.join(bin_full_labels_dir, label_name), arr=label)

                    # # Debug
                    # f, ax_arr = plt.subplots(1, 3)
                    # ax_arr[0].imshow(img)
                    # ax_arr[0].set_title("Image")
                    # ax_arr[1].imshow(label)
                    # ax_arr[1].set_title("Original Label")
                    # ax_arr[2].imshow(single_contour_label)
                    # ax_arr[2].set_title("Single Contour Label (Length {})".format(
                    #     len_single_contour))
                    #
                    # import pdb
                    # pdb.set_trace()

                    bin_pixel_count += len_single_contour
                    bin_img_count += 1

        images_per_bin_arr[bin_idx] = bin_img_count
        pixels_per_bin_arr[bin_idx] = bin_pixel_count

        print("Bin {}: Lengths [{}, {}], Num images {}, Num Pixels {}. Time {}".format(
            bin_idx, min_len, max_len, bin_img_count, bin_pixel_count,
            datetime.now() - bin_start_time))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    # Write a summary File
    summary_file = os.path.join(data_store_dir, 'summary.txt')
    file_handle = open(summary_file, 'w+')

    file_handle.write("Data Set Parameters {}\n".format('-' * 60))
    print("Script Duration {}".format(datetime.now() - script_start_time), file=file_handle)
    file_handle.write("Source : {}\n".format(img_dir))
    file_handle.write("Dataset Type: {}\n".format(data_set_type))
    file_handle.write("Random seed : {}\n".format(random_seed))
    file_handle.write("Lengths bins : {}\n".format(contour_lengths_bins))
    file_handle.write("Min pixels per bin : {}\n".format(min_pixels_per_bin))
    file_handle.write("{}\n".format('-'*80))

    for bin_idx, bin_len in enumerate(contour_lengths_bins):
        print("Bin [{}], Len {}, Num Images {}, Num Pixels {}".format(
            bin_idx, bin_len, images_per_bin_arr[bin_idx],
            pixels_per_bin_arr[bin_idx]), file=file_handle)

    file_handle.close()

    input("Press any Key to End")
    pdb.set_trace()
