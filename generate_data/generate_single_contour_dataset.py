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

if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7

    input_data_imgs_dir = './data/BIPED/edges/imgs/test/rgbr'
    input_data_labels_dir = './data/BIPED/edges/edge_maps/test/rgbr'

    data_store_dir = './data/single_contour_natural_images'

    contour_lengths_bins = [20, 50, 100, 150, 200]

    script_start_time = datetime.now()

    min_pixels_per_bin = 25000

    # Immutable
    # --------------------------------------
    plt.ion()
    np.random.seed(random_seed)

    # validate start data dir
    list_of_imgs = sorted(os.listdir(input_data_imgs_dir))
    list_of_labels = sorted(os.listdir(input_data_labels_dir))

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
        max_len = None
        bin_start_time = datetime.now()

        if bin_idx < len(contour_lengths_bins) - 1:
            max_len = contour_lengths_bins[bin_idx + 1]

        print("Generating Images with Contour length in [{}, {}]".format(min_len, max_len))

        # Create the bin data store directories
        bin_imgs_dir = \
            os.path.join(data_store_dir, 'len_{}_{}'.format(bin_len, max_len), 'images')
        bin_labels_dir = \
            os.path.join(data_store_dir, 'len_{}-{}'.format(bin_len, max_len), 'labels')
        if not os.path.exists(bin_imgs_dir):
            os.makedirs(bin_imgs_dir)
        if not os.path.exists(bin_labels_dir):
            os.makedirs(bin_labels_dir)

        # Randomly choose an image from the dataset to store the labels
        data_idx = np.random.randint(0, len(list_of_imgs))

        bin_pixel_count = 0
        bin_img_count = 0

        while bin_pixel_count <= min_pixels_per_bin:

            # Randomly choose an image from the start database to search for contours
            data_idx = np.random.randint(0, len(list_of_imgs))

            img_file = os.path.join(input_data_imgs_dir, list_of_imgs[data_idx])
            label_file = os.path.join(input_data_labels_dir, list_of_labels[data_idx])

            img = Image.open(img_file).convert('RGB')
            img = img.resize((256, 256))
            img = np.array(img)

            label = Image.open(label_file).convert('L')  # [0, 1] Mask
            label = label.resize((256, 256), Image.BILINEAR)  # Need for continuous edges
            label = np.array(label) / 255.0
            label[label > 0.1] = 1  # needed because of interpolation

            # Find a single contour with length in the bin
            single_contour = contour.get_random_contour(
                label, min_contour_len=min_len, max_contour_len=max_len)
            len_single_contour = len(single_contour)

            if len_single_contour > 0:
                # print("Contour of Length {} Found. Bin pixels count {}".format(
                #     len_single_contour, bin_pixel_count))

                # Create Single contour label
                single_contour_label = np.zeros_like(label)
                for point in single_contour:
                    single_contour_label[point[0], point[1]] = 1

                # save the image and label
                img_name = 'img_{}_clen_{}_'.format(bin_img_count, len_single_contour) +\
                    list_of_imgs[data_idx]
                label_name = 'img_{}_clen_{}_'.format(bin_img_count, len_single_contour) +\
                    list_of_labels[data_idx]

                plt.imsave(fname=os.path.join(bin_imgs_dir, img_name), arr=img)
                plt.imsave(fname=os.path.join(bin_labels_dir, label_name), arr=single_contour_label)

                # # Debug
                # f, ax_arr = plt.subplots(1, 3)
                # ax_arr[0].imshow(img)
                # ax_arr[0].set_title("Image")
                # ax_arr[1].imshow(label)
                # ax_arr[1].set_title("Original Label")
                # ax_arr[2].imshow(single_contour_label)
                # ax_arr[2].set_title("Single Contour Label (Length {})".format(len_single_contour))

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
    print("Script Duration {}".format(datetime.now() - script_start_time))

    for bin_idx, bin_len in enumerate(contour_lengths_bins):
        print("Bin [{}], Len {}, Num Images {}, Num Pixels {}".format(
            bin_idx, bin_len, images_per_bin_arr[bin_idx], pixels_per_bin_arr[bin_idx]))

    input("Press any Key to End")
    pdb.set_trace()
