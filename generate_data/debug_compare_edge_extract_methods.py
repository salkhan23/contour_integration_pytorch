import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import io
from skimage import transform as sk_transforms
from skimage import feature as sk_features
from skimage import filters
from skimage import color

from PIL import Image

img_size = (256, 256)
canny_edge_extract_sigma = 1.0

# data_dir = './data/edge_detection_data_set/val/images'
data_dir = './data/bsds/test'

plt.ion()

image_files = os.listdir(data_dir)

for img_file in sorted(image_files):

    img_file = os.path.join(data_dir, img_file)
    print(img_file)

    input_img = io.imread(img_file)
    resize_img_color = sk_transforms.resize(input_img, output_shape=img_size)
    resize_img = color.rgb2gray(resize_img_color)

    # Set low and high thresholds as a function of img median
    # Ref: http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
    img_median = np.median(resize_img)

    percent_edges = 0
    use_sigma = canny_edge_extract_sigma
    n_iters = 0

    lower_thresh = 0.03
    high_thresh = 0.08

    while not (lower_thresh < percent_edges < high_thresh):
        edge_img_canny = sk_features.canny(
            resize_img,
            sigma=use_sigma,
            low_threshold=0.4 * img_median,
            high_threshold=1.0 * img_median,
            use_quantiles=False,
        )

        percent_edges = \
            np.count_nonzero(edge_img_canny) / (resize_img.shape[0] * resize_img.shape[1])
        print("% edges in Image = {}".format(percent_edges))
        if percent_edges < lower_thresh:
            use_sigma -= 0.1
        elif percent_edges > high_thresh:
            use_sigma += 0.1
        n_iters += 1

        print("[{}] percent edges {}. Updated sigma = {}".format(n_iters, percent_edges, use_sigma))

        if n_iters == 15:
            break
        elif use_sigma < 0.1:
            break

    edge_img_sobel = filters.sobel(resize_img)

    f, ax_arr = plt.subplots(3, 3)

    # Show Original
    # ax_arr[0].imshow(input_img, cmap=plt.cm.gray)
    # ax_arr[0].set_title('Org - {}'.format(img))

    # Show Resized Original
    ax_arr[0][0].imshow(resize_img_color)
    ax_arr[0][0].set_title("resized original {}".format(img_size))

    # Edge Image
    ax_arr[0][1].imshow(edge_img_canny, cmap=plt.cm.gray)
    ax_arr[0][1].set_title("Edges- Canny")

    ax_arr[0][2].imshow(edge_img_sobel, cmap=plt.cm.gray)
    ax_arr[0][2].set_title("Edges- Sobel")

    # save the images
    plt.imsave(fname='canny_label.png', arr=edge_img_canny, cmap=plt.cm.gray)
    plt.imsave(fname='sobel_label.png', arr=edge_img_sobel, cmap=plt.cm.gray)

    # Images loaded as is
    target_canny = Image.open('canny_label.png').convert("L")
    target_sobel = Image.open('sobel_label.png').convert("L")

    ax_arr[1][0].axis('off')
    ax_arr[1][1].imshow(target_canny)
    ax_arr[1][1].set_title("Canny as is")
    ax_arr[1][2].imshow(target_sobel)
    ax_arr[1][2].set_title("Edge as is")

    # Load Images as Binary masks
    target_canny_map = Image.open('canny_label.png').convert("1")
    target_sobel_map = Image.open('sobel_label.png').convert("1")

    ax_arr[2][0].axis('off')
    ax_arr[2][1].imshow(target_canny_map)
    ax_arr[2][1].set_title("Canny binary mask")
    ax_arr[2][2].imshow(target_sobel_map)
    ax_arr[2][2].set_title("Sobel binary mask")

    import pdb

    pdb.set_trace()
    plt.close(f)

# ---------------------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------------------
import pdb

pdb.set_trace()
