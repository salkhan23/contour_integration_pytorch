# ---------------------------------------------------------------------------------------
# This is a debug script for checking responses to gabors fragments.
# Two possible starting points (for defining gabors)
#    (1) Specify Gabor parameters (Function: define_gabor_parameters)
#    (2) Specify the fragment (Function: define Gabor Fragment)
#
#  Given gabor parameters,
#   [1] Construct a test contour image (centered contour)
#   [2] Get/plot the responses of all centered neurons of the Fist /edge extracting
#   layer of a model
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import alexnet
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gabor_fits
import fields1993_stimuli


def define_gabor_parameters(frag_size):
    """

    :return:
    """
    bg_value = 0
    gabor_params_list = np.array([
        [0, -1., 145, 0.33, 2.00, 15.25, 0, 0]
    ])

    # ---------------------------------
    g_params = gabor_fits.convert_gabor_params_list_to_dict(gabor_params_list)
    frag = gabor_fits.get_gabor_fragment(g_params, frag_size)

    # Display Fragment
    plt.figure()
    plt.imshow(frag)
    plt.title("Generated Fragment")

    return frag, g_params, bg_value


def define_gabor_fragment(frag_size):
    """
     Explicitly Define Fragment.
     A Gabor Fit will be found.

    :param frag_size:
    :return:
    """
    bg_value = 0

    # frag = np.ones(frag_size, dtype='uint8') * 255
    # frag[:, frag_size[0] // 2 - 2, :] = 0
    # frag[:, frag_size[0] // 2 - 1, :] = 0
    # frag[:, frag_size[0] // 2, :] = 0
    # frag[:, frag_size[0] // 2 + 1, :] = 0
    # frag[:, frag_size[0] // 2 + 2, :] = 0

    frag = np.array([
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255],
        [255, 255, 0, 0, 0, 255, 255]
    ])
    frag = np.stack([frag, frag, frag], axis=-1)

    # --------------------------------------------------------------
    plt.figure()
    plt.imshow(frag)
    plt.title("Specified Fragment")
    import pdb
    pdb.set_trace()

    print("Finding Gabor Fit ...")
    frag = (frag - frag.min()) / (frag.max() - frag.min())
    gabor_params_list = gabor_fits.find_best_fit_2d_gabor(frag, verbose=1)

    g_params = gabor_fits.convert_gabor_params_list_to_dict(gabor_params_list)
    g_params.print_params(g_params)

    fitted_gabor = gabor_fits.get_gabor_fragment(gabor_params, frag_size[:2])

    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].imshow(frag)
    ax_arr[0].set_title("Specified Fragment")
    ax_arr[1].imshow(fitted_gabor)
    ax_arr[1].set_title("Generated Fragment")

    return fitted_gabor, g_params, bg_value


edge_extract_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


if __name__ == "__main__":
    random_seed = 20
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    fragment_size = (7, 7)
    full_tile_size = np.array([14, 14])
    image_size = np.array([256, 256, 3])

    # -----------------------------------------------------------------------------------
    #  Gabor Params
    # -----------------------------------------------------------------------------------
    fragment, gabor_params, bg = define_gabor_parameters(fragment_size)
    # fragment, gabor_params, bg = define_gabor_fragment()

    # -----------------------------------------------------------------------------------
    #  Test Image
    # -----------------------------------------------------------------------------------
    contour_len = 9
    beta_rotation = 15
    alpha_rotation = 0

    image, image_label = fields1993_stimuli.generate_contour_image(
        frag=fragment,
        frag_params=gabor_params,
        c_len=contour_len,
        beta=beta_rotation,
        alpha=alpha_rotation,
        f_tile_size=full_tile_size,
        img_size=image_size,
        random_alpha_rot=True,
        rand_inter_frag_direction_change=True,
        use_d_jitter=True,
        bg_frag_relocate=False,
        bg=bg
    )
    print(image_label)
    print("Label is valid? {}".format(fields1993_stimuli.is_label_valid(image_label)))

    plt.figure()
    plt.imshow(image)
    plt.title("Input Image")

    label_image = fields1993_stimuli.plot_label_on_image(
        image, image_label, full_tile_size, edge_color=(250, 0, 0), edge_width=2, display_figure=False)

    # Highlight Tiles
    full_tile_starts = fields1993_stimuli.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=image_size[1],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=image_size[0] // 2 - (full_tile_size[0] // 2)
    )

    label_image = fields1993_stimuli.highlight_tiles(
        label_image, full_tile_size, full_tile_starts, edge_color=(255, 255, 0))

    plt.figure()
    plt.imshow(label_image)
    plt.title("Labelled Image")

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    net = alexnet(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    net.features[0].register_forward_hook(edge_extract_cb)

    # -----------------------------------------------------------------------------------
    #  Get Center Neuron Activations
    # -----------------------------------------------------------------------------------
    # Pre-process image
    # ------------------
    test_image = (image - image.min()) / (image.max() - image.min())  # [0,1] Pixel Range
    # Normalize using ImageNet mean/std
    input_image = (test_image - np.array([[0.4208942, 0.4208942, 0.4208942]])) / \
        np.array([0.15286704, 0.15286704, 0.15286704])

    input_image = torch.from_numpy(np.transpose(input_image, axes=(2, 0, 1)))
    input_image = input_image.to(device)
    input_image = input_image.float()
    input_image = input_image.unsqueeze(dim=0)

    edge_extract_act = 0
    label_out = net(input_image)

    center_neuron_extract_out = \
        edge_extract_act[0, :, edge_extract_act.shape[2] // 2, edge_extract_act.shape[3] // 2]

    max_active_neuron = np.argmax(center_neuron_extract_out)
    string = "Max Active Neuron {}. Value={:0.2f}.".format(
        max_active_neuron, center_neuron_extract_out[max_active_neuron])

    print(string)

    # Plot Center Neuron Activations
    plt.figure()
    plt.plot(center_neuron_extract_out)
    plt.title(string)
    plt.xlabel("Channel Index")

    import pdb
    pdb.set_trace()
