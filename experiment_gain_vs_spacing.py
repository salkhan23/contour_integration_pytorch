# ---------------------------------------------------------------------------------------
# Li 2006 - Contour Gain vs Spacing Experiment
#
# Currently IoU is not calculated. Only gain is considered
#
# ---------------------------------------------------------------------------------------
import numpy as np
import os
from datetime import datetime
import copy

import torch
import torchvision.transforms.functional as transform_functional
from torchvision import transforms

import models.new_piech_models as new_piech_models
import gabor_fits
import fields1993_stimuli
import utils

import matplotlib.pyplot as plt   # for viewing images

edge_extract_act = np.array([])
cont_int_in_act = np.array([])
cont_int_out_act = np.array([])

INVALID_RESULT = -1000

# ---------------------------------------------------------------------------------------
# Base Gabor Parameters
#
# The following list defines a 10 sets of gabor parameters that form good contour
# fragments (visually). The optimal stimulus for a kernel, will be modified version of
# one of these param sets. Currently only orientation is modified. Each gabor set is
# itself a list describing gabor parameters for each channel. The parameters are
# [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma].
# However, functions from gabor_fits are used to convert to a more useable dictionary
# format
base_gabor_params_list = [
    [
        [0, -1.03, 27, 0.33, 1.00, 8.25, 0, 0]
    ],
    [
        [1.33, 0, 70, 1.10, 1.47, 9.25, 0, 0]
    ],
    [
        [0, 0, 150, 0.3, 0.86, 11.60, 0.61, 0]
    ],
    [
        [0, 0, 0, 0.3, 1.5, 7.2, 0.28, 0]
    ],
    [
        [0, 0, 160, -0.33, 0.80, 20.19, 0, 0]
    ],
    [
        [0.00, 0.00, -74, 0.29, 1.30, 4.17, 0.73, 0.15],
        [0.00, 0.00, -74, 0.53, 1.30, 4.74, 0.81, 0.17],
        [0.00, 0.00, -74, 0.27, 1.30, 4.39, 1.03, 0.17],
    ],
    [
        [0, -1.03, 135, 0.33, 1.0, 8.25, 0, 0]
    ],
    [
        [0.00, 0.00, 80, -0.33, 0.80, 20.19, 0, 0.00]
    ],
    [
        [0, 0, 120, 0.3, 0.86, 8.60, -0.61, 0]
    ],
    [
        [0, 0, -45, -0.46, 0.9, 25, 1, 0]
    ]
]

# Each gabor param set also defines a bg value which blends in the gabor to the rest of the image.
base_gabor_bg_list = [0, 0, 0, 0, 255, None, 0, 255, 0, 255]

base_gabor_parameters = []
for set_idx, gabor_set in enumerate(base_gabor_params_list):
    params = gabor_fits.convert_gabor_params_list_to_dict(gabor_set)

    for chan_params in params:
        chan_params['bg'] = base_gabor_bg_list[set_idx]

    base_gabor_parameters.append(params)

# ---------------------------------------------------------------------------------------


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at Edge Extract layer
        Callback to Retrieve the activations output of edge Extract layer
    """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer
        Callback to Retrieve the activations input & output of the contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


def process_image(
        model, devise_to_use, ch_mus, ch_sigmas, in_img, in_img_label=None, detect_thres=0.5):
    """
    Pass image through model and get iou score of the prediction if in_img_label is not None

    :param detect_thres:
    :param in_img_label:
    :param model:
    :param devise_to_use:
    :param in_img:
    :param ch_mus:
    :param ch_sigmas:
    :return:
    """
    # Zero all collected variables
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    edge_extract_act = 0
    cont_int_in_act = 0
    cont_int_out_act = 0

    normalize = transforms.Normalize(mean=ch_mus, std=ch_sigmas)
    model_in_img = normalize(in_img)
    model_in_img = model_in_img.to(devise_to_use).unsqueeze(0)

    # Pass the image through the model
    model.eval()

    if isinstance(model, new_piech_models.JointPathfinderContourResnet50):
        # Output is (contour_dataset_out, pathfinder_out)
        label_out, _ = model(model_in_img)
    else:
        label_out = model(model_in_img)

    iou = None
    if in_img_label is not None:
        in_img_label = in_img_label.to(devise_to_use).unsqueeze(0)

        preds = (torch.sigmoid(label_out) > detect_thres)

        iou = utils.intersection_over_union(preds.float(), in_img_label.float())
        iou = iou.cpu().detach().numpy()

    return iou


def find_optimal_stimulus(
        model, device_to_use, k_idx, ch_mus, ch_sigmas, extract_point, frag_size=np.array([7, 7]),
        img_size=np.array([256, 256, 3])):
    """

    :return:
    """
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    orient_arr = np.arange(0, 180, 5)

    img_center = img_size[0:2] // 2

    tgt_n_acts = np.zeros((len(base_gabor_parameters), len(orient_arr)))
    tgt_n_max_act = 0
    tgt_n_opt_params = None

    for base_gp_idx, base_gabor_params in enumerate(base_gabor_parameters):
        print("Processing Base Gabor Param Set {}".format(base_gp_idx))
        for o_idx, orient in enumerate(orient_arr):

            # Change orientation
            g_params = copy.deepcopy(base_gabor_params)
            for c_idx in range(len(g_params)):
                g_params[c_idx]["theta_deg"] = orient

            # Create Test Image - Single fragment @ center
            frag = gabor_fits.get_gabor_fragment(g_params, spatial_size=frag_size)
            bg = base_gabor_params[0]['bg']
            if bg is None:
                bg = fields1993_stimuli.get_mean_pixel_value_at_boundary(frag)

            test_img = np.ones(img_size, dtype='uint8') * bg

            add_one = 1
            if frag_size[0] % 2 == 0:
                add_one = 0

            test_img[
                img_center[0] - frag_size[0] // 2: img_center[0] + frag_size[0] // 2 + add_one,
                img_center[0] - frag_size[0] // 2: img_center[0] + frag_size[0] // 2 + add_one,
                :,
            ] = frag

            test_img = transform_functional.to_tensor(test_img)

            # # Debug - Show Test Image
            # plt.figure()
            # plt.imshow(np.transpose(test_img,axes=(1, 2, 0)))
            # plt.title("Input Image - Find optimal stimulus")
            # import pdb
            # pdb.set_trace()

            # Get target activations
            process_image(model, device_to_use, ch_mus, ch_sigmas, test_img)

            if extract_point == 'edge_extract_layer_out':
                center_n_acts = \
                    edge_extract_act[
                        0, :, edge_extract_act.shape[2]//2, edge_extract_act.shape[3]//2]
            elif extract_point == 'contour_integration_layer_in':
                center_n_acts = \
                    cont_int_in_act[
                        0, :, cont_int_in_act.shape[2]//2, cont_int_in_act.shape[3]//2]
            else:  # 'contour_integration_layer_out'
                center_n_acts = \
                    cont_int_out_act[
                        0, :, cont_int_out_act.shape[2]//2, cont_int_out_act.shape[3]//2]

            tgt_n_act = center_n_acts[k_idx]
            tgt_n_acts[base_gp_idx, o_idx] = tgt_n_act

            # # # Debug - Display all channel responses to individual test image
            # plt.figure()
            # plt.plot(center_n_acts)
            # plt.title("Center Neuron Activations. Base Gabor Set {}. Orientation {}".format(
            #     base_gp_idx, orient))

            if tgt_n_act > tgt_n_max_act:

                tgt_n_max_act = tgt_n_act
                tgt_n_opt_params = copy.deepcopy(g_params)

                max_active_n = int(np.argmax(center_n_acts))

                extra_info = {
                    'optim_stim_act_value': tgt_n_max_act,
                    'optim_stim_base_gabor_set': base_gp_idx,
                    'optim_stim_act_orient': orient,
                    'max_active_neuron_is_target': (max_active_n == k_idx),
                    'max_active_neuron_value': center_n_acts[max_active_n],
                    'max_active_neuron_idx': max_active_n

                }

                for item in tgt_n_opt_params:
                    item['extra_info'] = extra_info

        # # -----------------------------------------
        # # Debug - Tuning Curve for Individual base Gabor Params
        # plt.figure()
        # plt.plot(orient_arr, tgt_n_acts[base_gp_idx, :])
        # plt.title("Neuron {}: responses vs Orientation. Gabor Set {}".format(k_idx, base_gp_idx))

    # ---------------------------
    if tgt_n_opt_params is not None:

        # Save optimal tuning curve
        for item in tgt_n_opt_params:
            opt_base_g_params_set = item['extra_info']['optim_stim_base_gabor_set']
            item['extra_info']['orient_tuning_curve_x'] = orient_arr
            item['extra_info']['orient_tuning_curve_y'] = tgt_n_acts[opt_base_g_params_set, ]

        # # Debug: plot tuning curves for all gabor sets
        # # ------------------------------------------------
        # plt.figure()
        # for base_gp_idx, base_gabor_params in enumerate(base_gabor_parameters):
        #
        #     if base_gp_idx == tgt_n_opt_params[0]['extra_info']['optim_stim_base_gabor_set']:
        #         line_width = 5
        #         plt.plot(
        #             tgt_n_opt_params[0]['extra_info']['optim_stim_act_orient'],
        #             tgt_n_opt_params[0]['extra_info']['max_active_neuron_value'],
        #             marker='x', markersize=10,
        #             label='max active neuron Index {}'.format(
        #                 tgt_n_opt_params[0]['extra_info']['max_active_neuron_idx'])
        #         )
        #     else:
        #         line_width = 2
        #
        #     plt.plot(
        #         orient_arr, tgt_n_acts[base_gp_idx, ],
        #         label='param set {}'.format(base_gp_idx), linewidth=line_width
        #     )
        #
        # plt.legend()
        # plt.grid(True)
        # plt.title(
        #     "Kernel {}. Max Active Base Set {}. Is most responsive to this stimulus {}".format(
        #         k_idx,
        #         tgt_n_opt_params[0]['extra_info']['optim_stim_base_gabor_set'],
        #         tgt_n_opt_params[0]['extra_info']['max_active_neuron_is_target'])
        #     )
        #
        # import pdb
        # pdb.set_trace()

    return tgt_n_opt_params


def plot_tuning_curve(gp_params, k_idx=None):
    f = plt.figure()

    plt.title(
        "Kernel {}: Tuning Curve.\n Base Gabor Param set {}. Is Max Responsive Neuron {}.".format(
            k_idx,
            gp_params[0]['extra_info']['optim_stim_base_gabor_set'],
            gp_params[0]['extra_info']['max_active_neuron_is_target']))

    # Tuning Curve
    plt.plot(
        gp_params[0]['extra_info']['orient_tuning_curve_x'],
        gp_params[0]['extra_info']['orient_tuning_curve_y'],
        label='tuning curve')

    # # Highlight Max location
    # tgt_n_max_act = np.max(gp_params[0]['extra_info']['orient_tuning_curve_y'])
    # tgt_n_max_act_idx = np.argmax(gp_params[0]['extra_info']['orient_tuning_curve_y'])
    # plt.axvline(
    #     gp_params[0]['extra_info']['orient_tuning_curve_x'][tgt_n_max_act_idx], color='green')

    # Mark the value of the neuron which had the max response to this stimulus
    plt.plot(
        gp_params[0]['extra_info']['optim_stim_act_orient'],
        gp_params[0]['extra_info']['max_active_neuron_value'],
        marker='x',
        markersize=10,
        color='r',
        label='max_active_neuron: idx: {} value: {:0.2f}'.format(
            gp_params[0]['extra_info']['max_active_neuron_idx'],
            gp_params[0]['extra_info']['max_active_neuron_value']))

    plt.grid()
    plt.legend()
    plt.xlabel("Orientation")

    return f


def get_contour_gain_vs_spacing(
        model, device_to_use, g_params, k_idx, ch_mus, ch_sigmas, rslt_dir, full_tile_s_arr,
        frag_tile_s, c_len=7, n_images=50, img_size=np.array([256, 256, 3]), epsilon=1e-5):
    """
    TODO: Add description

    """
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    # tracking variables  -------------------------------------------------

    tgt_n = k_idx
    max_act_n_idx = g_params[0]['extra_info']['max_active_neuron_idx']

    tgt_n_out_acts = np.zeros((n_images, full_tile_s_arr.shape[0]))
    max_act_n_acts = np.zeros_like(tgt_n_out_acts)

    tgt_n_single_frag_acts = np.zeros(n_images)
    max_act_n_single_frag_acts = np.zeros_like(tgt_n_single_frag_acts)

    # -----------------------------------------------------------------
    frag = gabor_fits.get_gabor_fragment(g_params, spatial_size=frag_tile_s)
    bg = g_params[0]['bg']

    # First get response to Single fragment and co-linear distance = 1 (noise pattern)
    for img_idx in range(n_images):
        test_img, test_img_label, contour_frags_starts, end_acc_angle, start_acc_angle = \
            fields1993_stimuli.generate_contour_image(
                frag=frag,
                frag_params=g_params,
                c_len=1,
                beta=0,
                alpha=0,
                f_tile_size=np.array([14, 14]),
                img_size=img_size,
                random_alpha_rot=True,
                rand_inter_frag_direction_change=True,
                use_d_jitter=False,
                bg_frag_relocate=False,
                bg=bg
            )

        test_img = transform_functional.to_tensor(test_img)
        process_image(model, device_to_use, ch_mus, ch_sigmas, test_img)
        center_n_acts = \
            cont_int_out_act[
                0, :, cont_int_out_act.shape[2] // 2, cont_int_out_act.shape[3] // 2]

        tgt_n_single_frag_acts[img_idx] = center_n_acts[tgt_n]
        max_act_n_single_frag_acts[img_idx] = center_n_acts[max_act_n_idx]

    print("Tgt Neuron Single Fragment (RCD=1.0) Resp: mean {:0.2f}, std {:0.2f}".format(
        np.mean(tgt_n_single_frag_acts), np.std(tgt_n_single_frag_acts)))
    print("Max Active Neuron Single Fragment (RCD=1.0) Resp: mean {:0.2f}, std {:0.2f}".format(
        np.mean(max_act_n_single_frag_acts), np.std(max_act_n_single_frag_acts)))

    # # Debug
    # plt.figure()
    # plt.imshow(np.transpose(test_img, axes=(1, 2, 0)))
    # plt.title("Input Image")
    # import pdb
    # pdb.set_trace()

    for ft_idx, full_tile_s in enumerate(full_tile_s_arr):
        print("Processing Full Tile size = {}".format(full_tile_s))

        # Next Get responses for full_tile_s fragment spacing
        for img_idx in range(n_images):

            # (1) Create Test Image
            test_img, test_img_label, contour_frags_starts, end_acc_angle, start_acc_angle = \
                fields1993_stimuli.generate_contour_image(
                    frag=frag,
                    frag_params=g_params,
                    c_len=c_len,
                    beta=0,
                    alpha=0,
                    f_tile_size=full_tile_s,
                    img_size=img_size,
                    random_alpha_rot=True,
                    rand_inter_frag_direction_change=True,
                    use_d_jitter=False,
                    bg_frag_relocate=True,
                    bg=bg
                )

            test_img = transform_functional.to_tensor(test_img)
            # test_img_label = torch.from_numpy(np.array(test_img_label)).unsqueeze(0)

            # # Debug - Plot Test Image
            # # ------------------------
            # if img_idx == 0:
            #     disp_img = np.transpose(test_img.numpy(), axes=(1, 2, 0))
            #     disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min()) * 255.
            #     disp_img = disp_img.astype('uint8')
            #     disp_label = test_img_label.numpy()
            #
            #     print(disp_label)
            #     print("Label is valid? {}".format(fields1993_stimuli.is_label_valid(disp_label)))
            #
            #     plt.figure()
            #     plt.imshow(disp_img)
            #     plt.title("Input Image. Full Tile Size = {}".format(full_tile_s))
            #
            #     # Highlight Label Tiles
            #     disp_label_image = fields1993_stimuli.plot_label_on_image(
            #         disp_img,
            #         disp_label,
            #         full_tile_s,
            #         edge_color=(250, 0, 0),
            #         edge_width=2,
            #         display_figure=False
            #     )
            #
            #     # Highlight All background Tiles
            #     full_tile_starts = fields1993_stimuli.get_background_tiles_locations(
            #         frag_len=full_tile_s[0],
            #         img_len=img_size[1],
            #         row_offset=0,
            #         space_bw_tiles=0,
            #         tgt_n_visual_rf_start=img_size[0] // 2 - (full_tile_s[0] // 2)
            #     )
            #
            #     disp_label_image = fields1993_stimuli.highlight_tiles(
            #         disp_label_image,
            #         full_tile_s,
            #         full_tile_starts,
            #         edge_color=(255, 255, 0))
            #
            #     plt.figure()
            #     plt.imshow(disp_label_image)
            #     plt.title("Labeled Image. Full Tile Size = {}".format(full_tile_s))

            # (2) Get output Activations
            _ = process_image(model, device_to_use, ch_mus, ch_sigmas, test_img)

            center_n_acts = \
                cont_int_out_act[
                    0, :, cont_int_out_act.shape[2] // 2, cont_int_out_act.shape[3] // 2]

            tgt_n_out_acts[img_idx, ft_idx] = center_n_acts[tgt_n]
            max_act_n_acts[img_idx, ft_idx] = center_n_acts[max_act_n_idx]

    # # ------------------
    # import pdb
    # pdb.set_trace()
    # plt.close('all')

    # -------------------------------------------
    # Gain
    # -------------------------------------------
    # In Li2006, Gain was defined as output of neuron / mean output to noise pattern
    # where the noise pattern was defined as optimal stimulus at center of RF and all
    # others fragments were random. This corresponds to resp c_len=x/ mean resp clen=1
    tgt_n_avg_noise_resp = np.mean(tgt_n_single_frag_acts)
    max_active_n_avg_noise_resp = np.mean(max_act_n_single_frag_acts)

    tgt_n_gains = tgt_n_out_acts / (tgt_n_avg_noise_resp + epsilon)
    max_active_n_gains = max_act_n_acts / (max_active_n_avg_noise_resp + epsilon)

    tgt_n_mean_gain_arr = np.mean(tgt_n_gains, axis=0)
    tgt_n_std_gain_arr = np.std(tgt_n_gains, axis=0)

    max_act_n_mean_gain_arr = np.mean(max_active_n_gains, axis=0)
    max_act_n_std_gain_arr = np.std(max_active_n_gains, axis=0)

    # -----------------------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------------------
    # Fragment spacing measured in Relative co-linear distance metric
    # Defined as the ratio distance between fragments / length of fragment
    rcd_arr = (full_tile_s_arr[:, 0] - frag_tile_s[0]) / frag_tile_s[0]

    # Gain vs Spacing
    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].errorbar(
        rcd_arr, tgt_n_mean_gain_arr, tgt_n_std_gain_arr, label='Target Neuron {}'.format(tgt_n))
    ax_arr[1].errorbar(
        rcd_arr, max_act_n_mean_gain_arr, max_act_n_std_gain_arr,
        label='Max Active Neuron {}'.format(max_act_n_idx))

    ax_arr[0].set_xlabel("Contour Spacing (Relative Colinear Distance)")
    ax_arr[1].set_xlabel("Contour Spacing (Relative Colinear Distance)")
    ax_arr[0].set_ylabel("Gain")
    ax_arr[1].set_ylabel("Gain")
    ax_arr[0].set_ylim(bottom=0)
    ax_arr[1].set_ylim(bottom=0)
    ax_arr[0].grid()
    ax_arr[1].grid()
    ax_arr[0].legend()
    ax_arr[1].legend()
    f.suptitle("Contour Gain Vs Spacing - Neuron {}".format(k_idx))
    f.savefig(os.path.join(rslt_dir, 'gain_vs_spacing.jpg'), format='jpg')
    plt.close(f)

    # Output Activations vs Spacing
    f = plt.figure()
    plt.errorbar(rcd_arr, np.mean(tgt_n_out_acts, axis=0), np.std(tgt_n_out_acts, axis=0),
                 label='target_neuron_{}'.format(tgt_n))
    plt.errorbar(rcd_arr, np.mean(max_act_n_acts, axis=0), np.std(max_act_n_acts, axis=0),
                 label='max_active_neuron_{}'.format(max_act_n_idx))

    plt.plot(rcd_arr[0], tgt_n_avg_noise_resp,
             marker='x', markersize=10, color='red', label='tgt_n_single_frag_resp')
    plt.plot(rcd_arr[0], max_active_n_avg_noise_resp,
             marker='x', markersize=10, color='green', label='max_active_n_single_frag_resp')

    plt.legend()
    plt.grid()
    plt.xlabel("Fragment spacing (Relative Co-Linear Distance)")
    plt.ylabel("Activations")
    plt.title("Output Activations")
    f.savefig(os.path.join(rslt_dir, 'output_activations_vs_spacing.jpg'), format='jpg')
    plt.close(f)

    # Save output Activations
    tgt_n_mean_out_acts = np.mean(tgt_n_out_acts, axis=0)
    tgt_n_std_out_acts = np.std(tgt_n_out_acts, axis=0)

    return None, tgt_n_mean_gain_arr, tgt_n_std_gain_arr, max_act_n_mean_gain_arr, \
        max_act_n_std_gain_arr, tgt_n_avg_noise_resp, max_active_n_avg_noise_resp, \
        tgt_n_mean_out_acts, tgt_n_std_out_acts


def get_averaged_results(iou_mat, gain_mu_mat, gain_std_mat, exclude_idxs):
    """
    Get channel population averages. Excluding Channels in exclude_idxs

    :param iou_mat:  [n_channels x n_c_lens] (Not Used)
    :param gain_mu_mat: [n_channels x n_c_lens]
    :param gain_std_mat: [n_channels x n_c_lens]
    :param exclude_idxs: Indices of channel results to exclude  to exclude.

    :return:
    """
    pop_iou = None
    pop_mean_gains = None
    pop_std_gains = None

    valid_idxs = [idx for idx in range(gain_mu_mat.shape[0]) if idx not in exclude_idxs]

    # filt_iou = iou_mat[valid_idxs, ]
    filt_gains_mu = gain_mu_mat[valid_idxs, ]
    filt_gains_std = gain_std_mat[valid_idxs, ]

    # if len(filt_iou) is not 0:
    #     pop_iou = np.mean(filt_iou, axis=0)

    if len(filt_gains_mu) != 0:
        pop_mean_gains = np.mean(filt_gains_mu, axis=0)

    # For Two RVs, X and Y
    # Given mu_x, mu_y, sigma_x, sigma_y
    # sigma (standard deviation) of X + Y = np.sqrt(sigma_x**2 + sigma_y**2)
    # This gives the standard deviation of the sum, of X+Y, to get the average variance
    # if all samples were from same RV, just average the summed variance. Then sqrt it to
    # get avg std
    # REF: https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    if len(filt_gains_std) != 0:
        sum_var = np.sum(filt_gains_std ** 2, axis=0)
        avg_var = sum_var / filt_gains_std.shape[0]
        pop_std_gains = np.sqrt(avg_var)

    return pop_iou, pop_mean_gains, pop_std_gains


def plot_gain_vs_fragment_spacing(
        rcd_arr, mu_gain_arr, sigma_gain_arr, store_dir, f_name, f_title=None):
    """
     Plot and Save Gain vs Contour Spacing

    :param f_title:
    :param f_name:
    :param rcd_arr:
    :param mu_gain_arr:
    :param sigma_gain_arr:
    :param store_dir:
    :return:
    """
    if mu_gain_arr is not None:
        f = plt.figure()
        plt.errorbar(rcd_arr, mu_gain_arr, sigma_gain_arr)
        plt.xlabel("Fragment spacing (Relative Colinear Distance)")
        plt.ylabel("Gain")
        plt.ylim(bottom=0)
        plt.grid(True)
        if f_title is not None:
            plt.title("{}".format(f_title))
        f.savefig(os.path.join(store_dir, '{}.jpg'.format(f_name)), format='jpg')
        plt.close()


def write_population_avg_results(iou_arr, mean_gain_arr, std_gain_arr, f_handle):
    if iou_arr is not None:
        f_handle.write(
            "Population IoU      : [" + ", ".join('{:0.4f}'.format(x) for x in iou_arr) + "]\n")
    if mean_gain_arr is not None:
        f_handle.write(
            "Population mean gain: [" + ", ".join('{:0.4f}'.format(x) for x in mean_gain_arr)
            + "]\n")
    if std_gain_arr is not None:
        f_handle.write(
            "Population std gain : [" + ", ".join('{:0.4f}'.format(x) for x in std_gain_arr)
            + "]\n")


def plot_individual_neurons_separately(
        mu_mat, spacing_arr, no_optim_stim_n, below_th_n, store_dir, f_name, f_title):

    tile_single_dim = int(np.ceil(np.sqrt(len(mu_mat))))

    f, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(9, 9))
    for ch_idx in range(len(mu_mat)):

        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        if ch_idx in no_optim_stim_n:
            ax_arr[r_idx, c_idx].annotate('NOS', (0.1, 0.5), xycoords='axes fraction', va='center')
            ax_arr[r_idx, c_idx].axis('off')
            continue
        elif ch_idx in below_th_n:
            ax_arr[r_idx, c_idx].annotate('BTH', (0.1, 0.5), xycoords='axes fraction', va='bottom')

        ax_arr[r_idx, c_idx].plot(spacing_arr, mu_mat[ch_idx, ])
        ax_arr[r_idx, c_idx].locator_params(axis='y', nbins=2)
        ax_arr[r_idx, c_idx].set_xticks([])

    if f_title is not None:
        plt.suptitle("{}".format(f_title))
    f.savefig(os.path.join(store_dir, '{}.jpg'.format(f_name)), format='jpg')
    plt.close()


def main(model, base_results_dir, optimal_stim_extract_point='contour_integration_layer_out',
         full_tile_size_arr=np.array(
             [[14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21]]),
         frag_size=np.array([7, 7]), embedded_layer_identifier=None, optimal_stim_dict=None, n_images=50):
    """

    :param n_images:
    :param embedded_layer_identifier:
    :param model:
    :param base_results_dir:
    :param optimal_stim_extract_point:  Find optimal Stimulus @ which point. Can be:
        'edge_extract_layer_out', 'contour_integration_layer_in', 'contour_integration_layer_out'
    :param full_tile_size_arr:
    :param frag_size
    :param optimal_stim_dict: List of optimal stimuli for each channel. If present will not find optimal
        stimulus for each channel but just use the list. [Default is it is not provided]

    :return: optimal_param_dict for each channel, indexed by key = channel
    """

    # Contour Data Set Normalization (channel_wise_optimal_full14_frag7)
    chan_means = np.array([0.46958107, 0.47102246, 0.46911009])
    chan_stds = np.array([0.46108359, 0.46187091, 0.46111096])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    valid_edge_extract_points = [
        'edge_extract_layer_out',
        'contour_integration_layer_in',
        'contour_integration_layer_out'
    ]
    if optimal_stim_extract_point.lower() not in valid_edge_extract_points:
        raise Exception("Invalid optimal Stimulus Extraction point {}. Must be one of {}".format(
            optimal_stim_extract_point, valid_edge_extract_points))

    # Register Callbacks
    if embedded_layer_identifier is None:
        model.edge_extract.register_forward_hook(edge_extract_cb)
        model.contour_integration_layer.register_forward_hook(contour_integration_cb)
        n_channels = model.edge_extract.weight.shape[0]
    else:
        embedded_layer_identifier.edge_extract.register_forward_hook(edge_extract_cb)
        embedded_layer_identifier.contour_integration_layer.register_forward_hook(
            contour_integration_cb)
        n_channels = embedded_layer_identifier.edge_extract.weight.shape[0]

    # Results Directory
    results_store_dir = os.path.join(
        base_results_dir, 'experiment_gain_vs_spacing_frag_size_{}_test'.format(frag_size))
    print("Results store directory: {}".format(results_store_dir))
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    individual_neuron_results_store_dir = os.path.join(results_store_dir, 'individual_neurons')
    if not os.path.exists(individual_neuron_results_store_dir):
        os.makedirs(individual_neuron_results_store_dir)

    # Track optimal stimuli for each channel
    tracked_optimal_stim_dict = {}

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    relative_colinear_dist_arr = (full_tile_size_arr[:, 0] - frag_size[0]) / frag_size[0]

    tgt_neuron_mean_gain_mat = \
        np.ones((n_channels, len(relative_colinear_dist_arr))) * INVALID_RESULT  # [n_channels, n_len]
    tgt_neuron_std_gain_mat = np.ones_like(tgt_neuron_mean_gain_mat) * INVALID_RESULT
    tgt_neuron_noise_resp_arr = np.ones(n_channels) * INVALID_RESULT  # [n_channels]
    tgt_neuron_mean_out_acts = np.ones_like(tgt_neuron_mean_gain_mat) * INVALID_RESULT
    tgt_neuron_std_out_acts = np.ones_like(tgt_neuron_mean_gain_mat) * INVALID_RESULT

    max_active_neuron_mean_gain_mat = \
        np.ones((n_channels, len(relative_colinear_dist_arr))) * INVALID_RESULT
    max_active_neuron_std_gain_mat = np.ones_like(max_active_neuron_mean_gain_mat) * INVALID_RESULT
    max_active_neuron_noise_resp_arr = np.ones(n_channels) * INVALID_RESULT  # [n_channels]

    iou_per_len_mat = \
        np.ones((n_channels, len(relative_colinear_dist_arr))) * INVALID_RESULT  # [n_channels, n_len]

    for ch_idx in range(n_channels):
        print("{0} processing channel {1} {0}".format("*" * 20, ch_idx))

        # (1) Find optimal stimulus
        # -----------------------------------
        if optimal_stim_dict is None:
            print(">>>> Finding optimal stimulus")
            gabor_params = find_optimal_stimulus(
                model=model,
                device_to_use=device,
                k_idx=ch_idx,
                extract_point=optimal_stim_extract_point,
                ch_mus=chan_means,
                ch_sigmas=chan_stds,
                frag_size=frag_size,
            )

        else:
            print(">>>> Using stored optimal stimulus")
            gabor_params = optimal_stim_dict.get(ch_idx, None)

        if gabor_params is None:
            print("No optimal stimulus for channel {}".format(ch_idx))
            continue

        # Optimal stimulus tracking
        tracked_optimal_stim_dict[ch_idx] = gabor_params

        # Save Tuning Curve and Gabor fit params:
        n_results_dir = \
            os.path.join(individual_neuron_results_store_dir, 'neuron_{}'.format(ch_idx))
        if not os.path.exists(n_results_dir):
            os.makedirs(n_results_dir)

        fig = plot_tuning_curve(gabor_params, ch_idx)
        fig.savefig(os.path.join(n_results_dir, 'tuning_curve.jpg'), format='jpg')
        plt.close(fig)

        summary_file = os.path.join(n_results_dir, 'gabor_params.txt')
        file_handle = open(summary_file, 'w+')
        for e_idx, entry in enumerate(gabor_params):
            file_handle.write("{0} Channel {1} {0}\n".format('*' * 20, e_idx))

            for key in sorted(entry):
                if key == 'extra_info':
                    file_handle.write("Extra_info\n")
                    for key2 in sorted(entry['extra_info']):
                        file_handle.write("{}: {}\n".format(key2, entry['extra_info'][key2]))
                else:
                    file_handle.write("{}: {}\n".format(key, entry[key]))

        file_handle.close()

        # (2) Find Contour Gain Vs Fragment Spacing Tuning Curves
        # --------------------------------------------------------
        print(">>>> Getting contour gain vs fragment spacing performance ")
        ious, tgt_mean_gains, tgt_std_gain, max_active_mean_gains, max_active_std_gains, \
            tgt_n_noise_resp, max_active_n_noise_resp, tgt_n_mean_out_acts, tgt_n_std_out_acts = \
            get_contour_gain_vs_spacing(
                model=model,
                device_to_use=device,
                g_params=gabor_params,
                k_idx=ch_idx,
                ch_mus=chan_means,
                ch_sigmas=chan_stds,
                rslt_dir=n_results_dir,
                full_tile_s_arr=full_tile_size_arr,
                frag_tile_s=frag_size,
                c_len=7,
                n_images=n_images
            )

        tgt_neuron_mean_gain_mat[ch_idx, ] = tgt_mean_gains
        tgt_neuron_std_gain_mat[ch_idx, ] = tgt_std_gain
        tgt_neuron_noise_resp_arr[ch_idx] = tgt_n_noise_resp
        tgt_neuron_mean_out_acts[ch_idx, ] = tgt_n_mean_out_acts
        tgt_neuron_std_out_acts[ch_idx, ] = tgt_n_std_out_acts

        max_active_neuron_mean_gain_mat[ch_idx, ] = max_active_mean_gains
        max_active_neuron_std_gain_mat[ch_idx, ] = max_active_std_gains
        max_active_neuron_noise_resp_arr[ch_idx] = max_active_n_noise_resp

        iou_per_len_mat[ch_idx, ] = ious

    # -----------------------------------------------------------------------------------
    # Summary Results
    # -----------------------------------------------------------------------------------
    no_optim_stim_neurons = \
        [idx for idx, n_resp in enumerate(tgt_neuron_noise_resp_arr) if n_resp == INVALID_RESULT]

    print(">>>> Processing Results")
    summary_file = os.path.join(results_store_dir, 'results.txt')
    file_handle = open(summary_file, 'w+')

    file_handle.write("Experiment: Contour Gain vs. Fragment Spacing\n{}\n".format('-' * 80))
    file_handle.write("Full tile sizes: {}\n".format(full_tile_size_arr))
    rcd_string = ','.join('{:0.2f}'.format(item) for item in relative_colinear_dist_arr)
    file_handle.write("Fragment spacing (relative co-linear distance): {}\n".format(rcd_string))
    file_handle.write("Optimal Stimulus not found for {} neurons:\n {}\n".format(
        len(no_optim_stim_neurons), no_optim_stim_neurons))

    file_handle.write("\n\n{0} Summary Results {0} \n".format('-' * 20))

    # (1) Unfiltered  -------------------------------------------------------------------
    # Excluding neurons for which the optimal Stimulus was not found
    # Note: This may include unresponsive neurons (gains = 0 for all conditions)
    file_handle.write("{0} Unfiltered {0}\n".format('-' * 20))

    file_handle.write("Target Neurons\n")
    tgt_n_pop_iou, tgt_n_pop_mean_gain, tgt_pop_gain_std = get_averaged_results(
        iou_per_len_mat, tgt_neuron_mean_gain_mat, tgt_neuron_std_gain_mat, no_optim_stim_neurons)
    write_population_avg_results(
        tgt_n_pop_iou, tgt_n_pop_mean_gain, tgt_pop_gain_std, file_handle)

    file_handle.write("Max Active Neurons\n")
    max_active_n_pop_iou, max_active_n_pop_mean_gain, max_active_n_pop_gain_std = \
        get_averaged_results(
            iou_per_len_mat, max_active_neuron_mean_gain_mat, max_active_neuron_std_gain_mat,
            no_optim_stim_neurons)
    write_population_avg_results(
        max_active_n_pop_iou, max_active_n_pop_mean_gain, max_active_n_pop_gain_std, file_handle)

    #  (2) Filtered  --------------------------------------------------------------------
    # Excluding neurons for which the optimal Stimulus was not found
    # Additionally exclude neurons with noise responses less than 0.1. [Li -2006]: Neurons
    # that were not responsive to single bars or did not show a clear orientation tuning
    # preference were skipped. Method: Remove all neurons with a noise pattern
    # (single fragment) activation below a threshold
    min_clen_1_resp = 0.1
    file_handle.write(
        "{1} Filtered with c_len=1 response >= {0} {1}\n".format(min_clen_1_resp, '-' * 20))

    file_handle.write("Target Neurons\n")
    tgt_n_below_th_neurons = \
        [idx for idx, item in enumerate(tgt_neuron_noise_resp_arr) if item < min_clen_1_resp]
    file_handle.write("Removed {} neurons @ indexes {}\n".format(
        len(tgt_n_below_th_neurons), tgt_n_below_th_neurons))
    file_handle.write("Filtered results averaged over {} neurons\n".format(
        n_channels - len(tgt_n_below_th_neurons)))

    filt_tgt_n_pop_iou, filt_tgt_n_pop_mean_gain, filt_tgt_n_pop_gain_std = get_averaged_results(
        iou_per_len_mat, tgt_neuron_mean_gain_mat, tgt_neuron_std_gain_mat, tgt_n_below_th_neurons)
    write_population_avg_results(
        filt_tgt_n_pop_iou, filt_tgt_n_pop_mean_gain, filt_tgt_n_pop_gain_std, file_handle)

    file_handle.write("Max Active Neurons\n")
    max_active_below_th_neurons = \
        [idx for idx, item in enumerate(max_active_neuron_noise_resp_arr) if item < min_clen_1_resp]
    file_handle.write("Removed {} neurons @ indexes {}\n".format(
        len(max_active_below_th_neurons), max_active_below_th_neurons))
    file_handle.write("Filtered results averaged over {} neurons\n".format(
        n_channels - len(max_active_below_th_neurons)))

    filt_max_active_pop_iou, filt_max_active_pop_mean_gain, filt_max_active_pop_gain_std = \
        get_averaged_results(
            iou_per_len_mat, max_active_neuron_mean_gain_mat, max_active_neuron_std_gain_mat,
            max_active_below_th_neurons)
    write_population_avg_results(
        filt_max_active_pop_iou, filt_max_active_pop_mean_gain, filt_max_active_pop_gain_std,
        file_handle)

    # -----------------------------------------------------------------------------------
    # Detailed Results
    # -----------------------------------------------------------------------------------
    file_handle.write('\n\n{0} Detailed  Results {0}\n'.format("-" * 30))
    file_handle.write('Value -1000 = No Result\n')
    file_handle.write('{}\n'.format("-" * 80))

    np.set_printoptions(precision=3, linewidth=120)

    #  Target neuron
    file_handle.write("Target Neurons\n")
    print("Target Neuron mean Out Activations: \n" + 'np.' +
          repr(tgt_neuron_mean_out_acts), file=file_handle)
    print("Target Neuron std Out Activations: \n" + 'np.' +
          repr(tgt_neuron_std_out_acts), file=file_handle)
    print("Target Neuron Noise (single fragment) Response: \n" + 'np.' +
          repr(np.array(tgt_neuron_noise_resp_arr)), file=file_handle)
    print("Target Neuron Mean Gains: \n" + 'np.' +
          repr(tgt_neuron_mean_gain_mat), file=file_handle)
    print("Target Neuron Std Gains: \n" + 'np.' +
          repr(tgt_neuron_std_gain_mat), file=file_handle)

    # Max Active Neuron
    file_handle.write("Max Active Neurons\n")
    print("Max Active Neuron Noise (single fragment) Response: \n" + 'np.' +
          repr(np.array(max_active_neuron_noise_resp_arr)), file=file_handle)
    print("Max Active Neuron Mean Gains: \n" + 'np.' +
          repr(max_active_neuron_mean_gain_mat), file=file_handle)
    print("Max Active Neuron Std Gains: \n" + 'np.' +
          repr(max_active_neuron_std_gain_mat), file=file_handle)

    file_handle.close()

    # -----------------------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------------------
    plot_gain_vs_fragment_spacing(
        relative_colinear_dist_arr,
        tgt_n_pop_mean_gain,
        tgt_pop_gain_std,
        results_store_dir,
        f_name='tgt_n_pop_gain_vs_spacing_unfiltered',
        f_title='Target neuron population contour gain vs fragment spacing\n'
                '[Unfiltered]. Avg over {} neurons. Remove {} neurons - no optimal stimulus'.format(
                    n_channels - len(no_optim_stim_neurons), len(no_optim_stim_neurons))
    )

    # plot_gain_vs_fragment_spacing(
    #     relative_colinear_dist_arr,
    #     max_active_n_pop_mean_gain,
    #     max_active_n_pop_gain_std,
    #     results_store_dir,
    #     f_name='max_active_n_pop_gain_vs_spacing_unfiltered',
    #     f_title='Max Active neuron population contour gain vs fragment spacing\n'
    #             '[Unfiltered]. Avg over {} neurons. Remove {} neurons - no optimal stimulus'.format(
    #                 n_channels - len(no_optim_stim_neurons), len(no_optim_stim_neurons))
    # )

    plot_gain_vs_fragment_spacing(
        relative_colinear_dist_arr,
        filt_tgt_n_pop_mean_gain,
        filt_tgt_n_pop_gain_std,
        results_store_dir,
        f_name='tgt_n_pop_gain_vs_spacing_filtered',
        f_title='Target neuron population contour gain vs fragment spacing\n'
                '[filtered noise response > {}], Avg over {} neurons\n'
                'Removed {} neurons - No optimal stimulus {}, Below noise Th {}'.format(
                    min_clen_1_resp,
                    n_channels - len(tgt_n_below_th_neurons),
                    len(tgt_n_below_th_neurons),
                    len(no_optim_stim_neurons),
                    len(tgt_n_below_th_neurons) - len(no_optim_stim_neurons))
    )

    # -----------------------------------------------------------------------------------
    # Alternative Gain Calculation Method
    # Filter out anything with unreasonable gains, ie. with gains above a threshold
    # -----------------------------------------------------------------------------------
    max_gain = 20

    exclude_neurons = \
        [g_idx for g_idx, gains in enumerate(tgt_neuron_mean_gain_mat)
         if (np.any(gains >= max_gain) or np.all(gains == 0) or np.any(gains == INVALID_RESULT))]

    mg_filt_pop_iou, mg_filt_pop_mean_gain, mg_filt_pop_gain_std = get_averaged_results(
        iou_per_len_mat, tgt_neuron_mean_gain_mat, tgt_neuron_std_gain_mat, exclude_neurons)

    plot_gain_vs_fragment_spacing(
        relative_colinear_dist_arr,
        mg_filt_pop_mean_gain,
        mg_filt_pop_gain_std,
        results_store_dir,
        f_name='tgt_n_pop_gain_vs_spacing_filtered_max_gain',
        f_title='Target neuron population contour spacing vs length\n'
                '[max gain <= {}], Avg over {} neurons\n'
                'Removed {} neurons - No optimal stimulus {}, invalid gains {}'.format(
                    max_gain,
                    n_channels - len(exclude_neurons),
                    len(exclude_neurons),
                    len(no_optim_stim_neurons),
                    len(exclude_neurons) - len(no_optim_stim_neurons))
    )
    # -----------------------------------------------------------------------------------

    # plot_gain_vs_fragment_spacing(
    #     relative_colinear_dist_arr,
    #     filt_max_active_pop_mean_gain,
    #     filt_max_active_pop_gain_std,
    #     results_store_dir,
    #     f_name='max_active_n_pop_gain_vs_spacing_filtered',
    #     f_title='Max active neuron population contour gain vs fragment spacing\n'
    #             '[filtered noise response > {}], Avg over {} neurons\n'
    #             'Removed {} neurons - No optimal stimulus {}, Below noise Th {}'.format(
    #                 min_clen_1_resp,
    #                 n_channels - len(max_active_below_th_neurons),
    #                 len(max_active_below_th_neurons),
    #                 len(no_optim_stim_neurons),
    #                 len(max_active_below_th_neurons) - len(no_optim_stim_neurons))
    # )

    # Individual GAINS in a single tiled figure
    plot_individual_neurons_separately(
        mu_mat=tgt_neuron_mean_gain_mat,
        spacing_arr=relative_colinear_dist_arr,
        no_optim_stim_n=no_optim_stim_neurons,
        below_th_n=tgt_n_below_th_neurons,
        store_dir=results_store_dir,
        f_title='Target Neurons Individual Gains',
        f_name='tgt_n_indv_gain_vs_spacing'
    )

    # Individual OUTPUTS in a single tiled figure
    plot_individual_neurons_separately(
        mu_mat=tgt_neuron_mean_out_acts,
        spacing_arr=relative_colinear_dist_arr,
        no_optim_stim_n=no_optim_stim_neurons,
        below_th_n=tgt_n_below_th_neurons,
        store_dir=results_store_dir,
        f_title='Target Neurons Individual Outputs',
        f_name='tgt_n_indv_out_vs_spacing'
    )

    if optimal_stim_dict is not None:
        tracked_optimal_stim_dict = optimal_stim_dict

    return tracked_optimal_stim_dict


if __name__ == "__main__":
    random_seed = 10

    # Model
    # -------
    # Model trained with 5 iterations
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/new_model_resnet_based/Old/' \
        '/ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20200816_222302_baseline' \
        '/best_accuracy.pth'

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))
    results_dir = os.path.dirname(saved_model)

    main(net, results_dir)

    # -----------------------------------------------------------------------------------
    print("Running script took {}".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
