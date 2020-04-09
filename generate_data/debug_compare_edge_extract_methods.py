import matplotlib.pyplot as plt
import os

from skimage import io
from skimage import transform as sk_transforms
from skimage import feature as sk_features
from skimage import filters

from PIL import Image

img_size = (256, 256)
canny_edge_extract_sigma = 2.0

data_dir = './data/edge_detection_data_set/val/images'


plt.ion()

image_files = os.listdir(data_dir)

for img_file in sorted(image_files):

    img_file = os.path.join(data_dir, img_file)
    print(img_file)

    input_img = io.imread(img_file, as_gray=True)
    resize_img = sk_transforms.resize(input_img, output_shape=img_size)

    edge_img_canny = sk_features.canny(resize_img, sigma=canny_edge_extract_sigma)
    edge_img_sobel = filters.sobel(resize_img)

    f, ax_arr = plt.subplots(3, 3)

    # Show Original
    # ax_arr[0].imshow(input_img, cmap=plt.cm.gray)
    # ax_arr[0].set_title('Org - {}'.format(img))

    # Show Resized Original
    ax_arr[0][0].imshow(resize_img, cmap=plt.cm.gray)
    ax_arr[0][0].set_title("resized Original {}".format(img_size))

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
