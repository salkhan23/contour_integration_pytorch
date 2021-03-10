import numpy as np
import matplotlib.pyplot as plt
import torch
import fields1993_stimuli
import os


def get_2d_gaussian_kernel(shape, sigma=1.0):
    """
    Returns a 2d (unnormalized) Gaussian kernel of the specified shape.

    :param shape: x,y dimensions of the gaussian
    :param sigma: standard deviation of generated Gaussian
    :return:
    """
    ax = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    ay = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)
    # ax = np.linspace(-1, 1, shape[0])
    # ay = np.linspace(-1, 1, shape[1])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    kernel = kernel.reshape(shape)

    return kernel


def normalize_image(img):
    """
    Rescale image to [0,1] range for display using matplotlib
    :param img:
    :return:
    """
    img_max = np.max(img)
    img_min = np.min(img)

    new_image = np.copy(img)
    new_image = (new_image - img_min) / (img_max - img_min)

    return new_image


def view_ff_kernels(ff_kernels, results_store_dir=None):
    """
    Plot the feed forward kernels.
    [ch_out, ch_in=3, r, c]

    Kernels are normalized to range[0, 1] before display.

    :param results_store_dir:
    :param ff_kernels:
    :return:
    """
    n_ff_kernels = ff_kernels.shape[0]

    tile_single_dim = np.int(np.ceil(np.sqrt(n_ff_kernels)))
    fig, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(9, 9))
    fig.suptitle("Feed Forward Kernels")

    for ch_idx in range(n_ff_kernels):
        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        ff_kernel = np.transpose(ff_kernels[ch_idx, ], axes=(1, 2, 0))  # Channel first for display
        ff_kernel = (ff_kernel - ff_kernel.min()) / (ff_kernel.max() - ff_kernel.min())

        ax_arr[r_idx, c_idx].imshow(ff_kernel)
        ax_arr[r_idx, c_idx].axis('off')

    fig.tight_layout()

    if results_store_dir is not None:
        fig.savefig(os.path.join(results_store_dir, 'feedforward_kernels.jpg'), format='jpg')
        # plt.close(fig)

    return fig, ax_arr


def view_spatial_lateral_kernels(e_kernels, i_kernels, spatial_func=np.mean, results_store_dir=None):
    """
    Display each output kernel, with the sum of all inputs over the channel dimension.
     [ch_out, ch_in, r, c]

    spaital_func over all channels (function must change dimensions to 1). Also must have keyword axis.
    "sum, max, min, etc"

    :param spatial_func: [default: max]
    :param e_kernels:
    :param i_kernels:
    :param results_store_dir:
    :return:
    """
    n_out_ch = e_kernels.shape[0]  # Assumes channels of E and I kernels are the same.

    # -----------------------------------------------------------------------------------
    # View the lateral kernels
    # -----------------------------------------------------------------------------------
    tile_single_dim = np.int(np.ceil(np.sqrt(n_out_ch)))

    f_e, ax_arr_e = plt.subplots(tile_single_dim, tile_single_dim, figsize=(9, 9))
    f_e.suptitle("Excitatory Lateral Kernels - {} across input channels".format(spatial_func.__name__))

    f_i, ax_arr_i = plt.subplots(tile_single_dim, tile_single_dim, figsize=(9, 9))
    f_i.suptitle("Inhibitory Lateral Kernels - {} across input channels".format(spatial_func.__name__))

    for ch_idx in range(n_out_ch):
        e_idv_kernel = e_kernels[ch_idx, ]
        i_idv_kernel = i_kernels[ch_idx, ]

        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        ax_arr_e[r_idx, c_idx].imshow(spatial_func(e_idv_kernel, axis=0))
        ax_arr_e[r_idx, c_idx].axis('off')

        ax_arr_i[r_idx, c_idx].imshow(spatial_func(i_idv_kernel, axis=0))
        ax_arr_i[r_idx, c_idx].axis('off')

    f_e.tight_layout()
    f_i.tight_layout()

    if results_store_dir is not None:
        f_e.savefig(os.path.join(results_store_dir, 'lateral_excitatory_kernels.jpg'), format='jpg')
        # plt.close(f_e)
        f_i.savefig(os.path.join(results_store_dir, 'lateral_inhibitory_kernels.jpg'), format='jpg')
        # plt.close(f_i)

    return f_e, ax_arr_e, f_i, ax_arr_i


# ---------------------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------------------
def intersection_over_union(outputs, targets):
    """
    Calculates Jaccard Distance.

    REF:  https://github.com/arturml/pytorch-unet

    :param outputs:
    :param targets:
    :return:
    """
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)  # sum across images
    # -intersection to get rid of double values for correct preds
    union = (outputs + targets).sum(1) - intersection

    jac = (intersection + 0.001) / (union + 0.001)

    return jac.mean()


class PunctureImage(object):
    """
    This is an input transform
    Add random occlusion bubbles to image.
    REF: Gosselin and Schyns - 2001 - Bubbles: a technique to reveal the use of information in
         recognition tasks

    This is actually the opposite of the reference technique, instead of masking the image and
    then revealing parts of it through bubbles. Masked out gaussian bubbles are added to the image.

    :param n_bubbles: number of bubbles
    :param fwhm: bubble full width half magnitude. Note the actual tile size is 2 * fwhm
    :param peak_bubble_transparency: 0 = Fully opaque at center (default), 1= Fully visible
        at center (no occlusion at all)

    """

    def __init__(self, n_bubbles=0, fwhm=11, tile_size=None, peak_bubble_transparency=0):

        if 0 > peak_bubble_transparency or 1 < peak_bubble_transparency:
            raise Exception("Bubble transparency {}, should be between [0, 1]".format(
                peak_bubble_transparency))
        self.peak_bubble_transparency = peak_bubble_transparency

        self.n_bubbles = n_bubbles

        self.fwhm = fwhm  # full width half magnitude
        self.bubble_sigma = fwhm / 2.35482

        if tile_size is not None:
            self.tile_size = tile_size
        else:
            if isinstance(fwhm, np.ndarray):
                max_fwhm = max(fwhm)
                self.tile_size = np.array([np.int(2 * max_fwhm), np.int(2 * max_fwhm)])
            else:
                self.tile_size = np.array([np.int(2 * self.fwhm), np.int(2 * self.fwhm)])

    def __call__(self, img, start_loc_arr=None):
        """
        Args:
            img (Tensor) : Tensor image of size (C, H, W) to be normalized.
            start_loc_arr: Starting location of the full tile. If star location
            is specified it will only add as many bubbles as the length of the
            start_location array.

        Returns:
            Tensor: Normalized Tensor image.
        """
        _, h, w = img.shape

        img = img.permute(1, 2, 0)

        if start_loc_arr is not None:
            n_bubbles = len(start_loc_arr)
        else:
            n_bubbles = self.n_bubbles
            start_loc_arr = np.array([
                np.random.randint(h - self.tile_size[0], size=self.n_bubbles),
                np.random.randint(w - self.tile_size[1], size=self.n_bubbles),
            ]).T

        if isinstance(self.bubble_sigma, np.ndarray):
            sigma_arr = np.random.choice(self.bubble_sigma, size=n_bubbles)
        else:
            sigma_arr = np.ones(n_bubbles) * self.bubble_sigma

        mask = torch.ones_like(img)
        for start_loc_idx, start_loc in enumerate(start_loc_arr):
            bubble_frag = get_2d_gaussian_kernel(
                shape=self.tile_size, sigma=sigma_arr[start_loc_idx])
            bubble_frag = torch.from_numpy(bubble_frag)
            bubble_frag = bubble_frag.float().unsqueeze(-1)
            bubble_frag = 1 - bubble_frag

            mask = fields1993_stimuli.tile_image(
                mask,
                bubble_frag,
                start_loc,
                rotate_frags=False,
                gaussian_smoothing=False,
                replace=False
            )

        mask = mask + self.peak_bubble_transparency
        mask[mask > 1] = 1

        ch_means = img.mean(dim=[0, 1])  # Channel Last
        # print("Channel means  {}".format(ch_means))

        masked_img = mask * img + (1 - mask) * ch_means * torch.ones_like(img)

        # # Debug
        # import matplotlib.pyplot as plt
        # plt.ion()
        #
        # f, ax_arr = plt.subplots(1, 3)
        # ax_arr[0].imshow(img)
        # ax_arr[0].set_title("Original Image")
        # ax_arr[1].imshow(masked_img)
        # ax_arr[1].set_title("Punctured Image")
        # ax_arr[2].imshow(mask)
        # ax_arr[2].set_title("Mask")
        #
        # import pdb
        # pdb.set_trace()

        return masked_img.permute(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(n_bubbles={}, fwhm = {}, bubbles_sigma={}, tile_size={}), ' \
               'max bubble transparency={}'.format(
                   self.n_bubbles, self.fwhm, self.bubble_sigma, self.tile_size,
                   self.peak_bubble_transparency)
