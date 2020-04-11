# ---------------------------------------------------------------------------------------
# Create Edge only images (using Canny Edge Extractor) from  Imagenet Images
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from skimage import io
from skimage import transform as sk_transforms
from skimage import feature as sk_features
from skimage import color


IMAGENET_DIR = '/home/salman/workspace/keras/my_projects/contour_integration/data/imagenet-data'


def get_edge_image(in_img, sigma):
    """

    REF:
    # Set low and high thresholds as a function of img median
    # Ref: http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
    """

    img_median = np.median(in_img)

    use_sigma = sigma

    percent_edges = 0
    n_iters = 0

    lower_thresh = 0.03  # for percentages of edges in image
    high_thresh = 0.08

    while not (lower_thresh < percent_edges < high_thresh):
        edge_img_canny = sk_features.canny(
            in_img,
            sigma=use_sigma,
            low_threshold=0.4 * img_median,
            high_threshold=1.0 * img_median,
            use_quantiles=False,
        )

        percent_edges = \
            np.count_nonzero(edge_img_canny) / (in_img.shape[0] * in_img.shape[1])

        # print("% edges in Image = {}".format(percent_edges))
        if percent_edges < lower_thresh:
            use_sigma -= 0.1
        elif percent_edges > high_thresh:
            use_sigma += 0.1
        n_iters += 1

        # print("[{}] percent edges {}. Updated sigma = {}".format(
        #     n_iters, percent_edges, use_sigma))

        if n_iters == 15:
            break
        elif use_sigma < 0.1:
            break

    return edge_img_canny


def generate_data_set(
        process_set, n_img_per_cat, store_dir, img_size=(224, 224), canny_edge_extract_sigma=2.5):

    image_dir = os.path.join(IMAGENET_DIR, process_set)
    list_of_dir = [os.path.join(image_dir, d) for d in sorted(os.listdir(image_dir))]

    # Create Results Folders
    r_img_dir = os.path.join(store_dir, process_set, 'images')
    r_label_dir = os.path.join(store_dir, process_set, 'labels')
    if not os.path.exists(r_img_dir):
        os.makedirs(r_img_dir)
    if not os.path.exists(r_label_dir):
        os.makedirs(r_label_dir)

    for class_dir in list_of_dir:

        list_of_all_files = os.listdir(class_dir)

        # Randomly Select NUM_TRAIN_IMAGES_PER_CATEGORY images
        img_idx_arr = np.random.randint(0, len(list_of_all_files), size=n_img_per_cat)
        candidate_images = [list_of_all_files[img_idx] for img_idx in img_idx_arr]

        for img in candidate_images:
            img_file = os.path.join(class_dir, img)
            # print(img_file)

            input_img = io.imread(img_file)
            resize_img_color = sk_transforms.resize(input_img, output_shape=img_size)
            resize_img = color.rgb2gray(resize_img_color)

            edge_img = get_edge_image(resize_img, canny_edge_extract_sigma)

            # save the images
            plt.imsave(fname=os.path.join(r_img_dir, img), arr=resize_img_color,)
            plt.imsave(fname=os.path.join(r_label_dir, img), arr=edge_img, cmap=plt.cm.gray)

            # Debug
            # -----
            # f, ax_arr = plt.subplots(1, 2)

            # # Show Original
            # # ax_arr[0].imshow(input_img, cmap=plt.cm.gray)
            # # ax_arr[0].set_title('Org - {}'.format(img))
            #
            # # Show Resized Original
            # ax_arr[0].imshow(resize_img, cmap=plt.cm.gray)
            # ax_arr[0].set_title("resized Original {}".format(IMAGE_SIZE))
            #
            # # Edge Image
            # ax_arr[1].imshow(edge_img, cmap=plt.cm.gray)
            # ax_arr[1].set_title("Edges")


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    edge_extract_sigma = 1.0

    # data_set_dir = './data/edge_detection_data_set_canny_sigma_{}'.format(edge_extract_sigma)
    data_set_dir = './data/edge_detection_data_set_canny_dynamic'
    n_train_images_per_category = 50
    n_val_images_per_category = 2

    # -----------------------------------
    plt.ion()
    random_seed = 10
    np.random.seed(random_seed)

    start_time = datetime.now()

    print("Creating Natural Image Edge Detection data set @ {}".format(data_set_dir))

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    print(" Generating Train Images ....")
    generate_data_set(
        process_set='train',
        n_img_per_cat=n_train_images_per_category,
        store_dir=data_set_dir,
        canny_edge_extract_sigma=edge_extract_sigma
    )

    print('Generating Validation Images')
    generate_data_set(
        process_set='val',
        n_img_per_cat=n_val_images_per_category,
        store_dir=data_set_dir,
        canny_edge_extract_sigma=edge_extract_sigma
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("Processing Took: {}".format(datetime.now() - start_time))

    import pdb
    pdb.set_trace()
