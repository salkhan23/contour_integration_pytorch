# ---------------------------------------------------------------------------------------
# Track Dynamic behavior of neurons across  time steps
# Save them in a subdirectory where the model weights are stored
# ---------------------------------------------------------------------------------------
import numpy as np
import os
from datetime import datetime
import copy
import pickle

import torch
import torchvision.transforms.functional as transform_functional
from torchvision import transforms

import models.new_piech_models as new_piech_models
import gabor_fits
import fields1993_stimuli
import utils

import matplotlib.pyplot as plt   # for viewing images
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'lines.markeredgewidth': 3
})

# -------------------------------------------------------------------------------
# Support Functions - copied from experiment_gain_vs_len.py
# -------------------------------------------------------------------------------
edge_extract_act = np.array([])
cont_int_in_act = np.array([])
cont_int_out_act = np.array([])

INVALID_RESULT = -1000


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
        # Output is contour_dataset_out, pathfinder_out
        label_out, _ = model(model_in_img)
    else:
        label_out = model(model_in_img)

    iou = None
    preds = None
    if in_img_label is not None:
        in_img_label = in_img_label.to(devise_to_use).unsqueeze(0)

        preds = (torch.sigmoid(label_out) > detect_thres)

        iou = utils.intersection_over_union(preds.float(), in_img_label.float())
        iou = iou.cpu().detach().numpy()

        # Debug show predictions
        # z = preds.float().squeeze()
        # plt.figure()
        # plt.imshow(z)

    return iou, preds


def find_optimal_stimulus(
        model, device_to_use, k_idx, ch_mus, ch_sigmas, extract_point, frag_size=np.array([7, 7]),
        img_size=np.array([256, 256, 3])):
    """
    Copied from experiment gain vs len

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
            # # -----------------------
            # plt.figure()
            # plt.imshow(np.transpose(test_img, axes=(1, 2, 0)))
            # plt.title("Input Image - Find optimal stimulus")
            # import pdb
            # pdb.set_trace()

            # Get target activations
            process_image(model, device_to_use, ch_mus, ch_sigmas, test_img)

            # Get Target Neuron Activation
            # ----------------------------
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

            # # Debug - Display all channel responses to individual test image
            # # --------------------------------------------------------------
            # plt.figure()
            # plt.plot(center_n_acts)
            # plt.title("Center Neuron Activations. Base Gabor Set {}. Orientation {}".format(
            #     base_gp_idx, orient))
            # import pdb
            # pdb.set_trace()

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
        # import pdb
        # pdb.set_trace()

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
        # )
        #
        # import pdb
        # pdb.set_trace()

    return tgt_n_opt_params


def get_responses_per_iteration(
        model, device, frag, g_params, bg, c_len, n_images, full_tile_size, img_size, chan_means, chan_stds, ch_idx,):

    iou_arr = []
    e_act_arr = []
    i_act_arr = []

    for i_idx in range(n_images):

        # Generate the Image
        test_img, test_img_label, contour_frags_starts, end_acc_angle, start_acc_angle = \
            fields1993_stimuli.generate_contour_image(
                frag=frag,
                frag_params=g_params,
                c_len=c_len,
                beta=0,
                alpha=0,
                f_tile_size=full_tile_size,
                img_size=img_size,
                random_alpha_rot=True,
                rand_inter_frag_direction_change=True,
                use_d_jitter=False,
                bg_frag_relocate=True,
                bg=bg
            )

        test_img = transform_functional.to_tensor(test_img)
        test_img_label = torch.from_numpy(np.array(test_img_label)).unsqueeze(0)

        # Pass the image through the model
        iou, preds = process_image(model, device, chan_means, chan_stds, test_img, test_img_label)
        iou_arr.append(iou)

        # # Debug:  Display the image
        # if iou < 1:
        #     f, ax_arr = plt.subplots(1, 3)
        #     f.suptitle("IoU = {}".format(iou) )
        #
        #     disp_img = np.transpose(test_img.numpy(), axes=(1, 2, 0))
        #     disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min()) * 255
        #     disp_img = disp_img.astype('uint8')
        #     disp_label = test_img_label.numpy()
        #
        #     print(disp_label)
        #     print("Label is valid? {}".format(fields1993_stimuli.is_label_valid(disp_label)))
        #
        #     ax_arr[0].imshow(disp_img)
        #     ax_arr[0].set_title('input image')
        #
        #     # Highlight Label Tiles
        #     disp_label_image = fields1993_stimuli.plot_label_on_image(
        #         disp_img,
        #         disp_label,
        #         full_tile_size,
        #         edge_color=(250, 0, 0),
        #         edge_width=2,
        #         display_figure=False
        #     )
        #
        #     ax_arr[1].imshow(disp_label_image)
        #     ax_arr[1].set_title('Label')
        #
        #     preds = preds.detach().cpu().numpy()
        #     preds = np.squeeze(preds, axis=0)
        #     preds = preds.astype('uint8')
        #     disp_label_image = fields1993_stimuli.plot_label_on_image(
        #         disp_img,
        #         preds,
        #         full_tile_size,
        #         edge_color=(0, 250, 0),
        #         edge_width=2,
        #         display_figure=False
        #     )
        #
        #     ax_arr[2].imshow(disp_label_image)
        #     ax_arr[2].set_title('Predictions')
        #
        #     import pdb
        #     pdb.set_trace()

        # Get the activities for the central neurons
        e_recurr_act = model.contour_integration_layer.x_per_iteration  # activation Volume
        i_recurr_act = model.contour_integration_layer.y_per_iteration

        c_neuron_e_act = []
        c_neuron_i_act = []
        n_b, n_ch, n_row, n_col = e_recurr_act[0].shape

        for t_idx in range(model.contour_integration_layer.n_iters):
            c_neuron_e_act.append(
                e_recurr_act[t_idx][0, ch_idx, n_row // 2, n_col // 2].detach().cpu().numpy())
            c_neuron_i_act.append(
                i_recurr_act[t_idx][0, ch_idx, n_row // 2, n_col // 2].detach().cpu().numpy())

        e_act_arr.append(c_neuron_e_act)
        i_act_arr.append(c_neuron_i_act)

    iou_arr = np.array(iou_arr)
    e_act_arr = np.array(e_act_arr)
    i_act_arr = np.array(i_act_arr)

    return iou_arr.mean(), \
        np.mean(e_act_arr, axis=0), np.std(e_act_arr, axis=0), \
        np.mean(i_act_arr, axis=0), np.std(i_act_arr, axis=0),


# -------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------
def main(model, optimal_stim_dict=None, r_dir="."):

    # Contour Data Set Normalization (channel_wise_optimal_full14_frag7)
    chan_means = np.array([0.46958107, 0.47102246, 0.46911009])
    chan_stds = np.array([0.46108359, 0.46187091, 0.46111096])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    frag_size = np.array([7, 7])
    full_tile_size = np.array([14, 14])
    img_size = np.array([256, 256, 3])

    # Get temporal responses for  contour lengths
    c_len_arr = np.array([1, 3, 5, 7, 9])
    # c_len_arr = np.array([9])

    # Average responses over  n_images
    n_images = 20

    # -----------------------------------------------------------------------------------
    # Register Callbacks
    # -----------------------------------------------------------------------------------
    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)
    n_channels = model.edge_extract.weight.shape[0]

    # -----------------------------------------------------------------------------------
    # Find optimal stimuli for each kernel
    # -----------------------------------------------------------------------------------
    print(">>>> Getting Optimal stimuli for kernels...")
    if optimal_stim_dict is not None:
        tracked_optimal_stim_dict = optimal_stim_dict
        print("Use Stored Responses")

    else:
        tracked_optimal_stim_dict = {}

        for ch_idx in range(n_channels):
            print("{0} processing channel {1} {0}".format("*" * 20, ch_idx))

            gabor_params = find_optimal_stimulus(
                model=model,
                device_to_use=device,
                k_idx=ch_idx,
                extract_point='contour_integration_layer_out',
                ch_mus=chan_means,
                ch_sigmas=chan_stds,
                frag_size=frag_size,
            )

            tracked_optimal_stim_dict[ch_idx] = gabor_params

        # Save the optimal stimuli

        pickle_file = os.path.join(results_dir, 'optimal_stimuli.pickle')
        print("Saving optimal gabor parameters @ {}".format(pickle_file))

        with open(pickle_file, 'wb') as h:
            pickle.dump(tracked_optimal_stim_dict, h)

    # -----------------------------------------------------------------------------------
    # Get Dynamic time responses for each kernel
    # -----------------------------------------------------------------------------------
    print(">>>> Getting responses per time step")
    # Tell the model to store iterative predictions
    model.contour_integration_layer.store_recurrent_acts = True

    # # Increase the number of time steps to see what happens beyond trained iterations
    train_n_iter = model.contour_integration_layer.n_iters - 1
    # model.contour_integration_layer.n_iters = 7

    overall_results = {}
    # results is dictionary of dictionaries (referenced by channel index), one for each channel
    # each channel dictionary contains
    # {
    #       iou_per_len,
    #       for each Length
    #           mean_clen_1_e_act,
    #           std_clen_1_e_act,
    #           mean_clea_1_i_act,
    #           std_c_len_1_i_act
    # }

    for ch_idx in range(n_channels):
        print("{0} processing channel {1} {0}".format("*" * 20, ch_idx))

        g_params = tracked_optimal_stim_dict.get(ch_idx, None)

        if g_params is not None:

            frag = gabor_fits.get_gabor_fragment(g_params, spatial_size=frag_size)
            bg = g_params[0]['bg']

            iou_per_len_arr = []
            ch_results_dict = {}

            for c_len in c_len_arr:

                print("length {}".format(c_len))

                iou, mean_e_resp, std_e_resp, mean_i_resp, std_i_resp = get_responses_per_iteration(
                    model=model,
                    device=device,
                    g_params=g_params,
                    frag=frag,
                    bg=bg,
                    n_images=n_images,
                    c_len=c_len,
                    full_tile_size=full_tile_size,
                    img_size=img_size,
                    chan_means=chan_means,
                    chan_stds=chan_stds,
                    ch_idx=ch_idx
                )

                iou_per_len_arr.append(iou)
                ch_results_dict['c_len_{}_mean_e_resp'.format(c_len)] = mean_e_resp
                ch_results_dict['c_len_{}_std_e_resp'.format(c_len)] = std_e_resp
                ch_results_dict['c_len_{}_mean_i_resp'.format(c_len)] = mean_i_resp
                ch_results_dict['c_len_{}_std_i_resp'.format(c_len)] = std_i_resp

            ch_results_dict['iou_per_len'] = np.array(iou_per_len_arr)
            overall_results[ch_idx] = ch_results_dict

    # ----------------------------------------------------------------------------
    # Plot the results
    # ----------------------------------------------------------------------------
    print("Plotting Results ...")

    for ch_idx in range(n_channels):

        ch_results = overall_results.get(ch_idx, None)

        if ch_results is not None:

            per_chan_r_dir = os.path.join(r_dir, 'individual_channels/{}'.format(ch_idx))
            if not os.path.exists(per_chan_r_dir):
                os.makedirs(per_chan_r_dir)

            f_iou = plt.figure(figsize=(9, 9))
            plt.plot(c_len_arr, ch_results['iou_per_len'], marker='x')
            plt.title("IoU per length. Channel {}".format(ch_idx))
            plt.xlabel("Length")
            plt.ylabel("IoU")
            f_iou.savefig(os.path.join(per_chan_r_dir, 'iou_ch_{}.jpg'.format(ch_idx)), format='jpg')

            f_resp, ax_arr = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            f_resp.suptitle("Responses per Iteration. Channel ={}".format(ch_idx))

            ax_arr[0].set_title("Excitatory")
            # ax_arr[0].set_xlabel("Time")
            ax_arr[0].set_ylabel("Activation")
            ax_arr[0].axvline(train_n_iter, linestyle='--', color='black')

            ax_arr[1].set_title("Inhibitory")
            ax_arr[1].set_xlabel("time")
            ax_arr[1].set_ylabel("Activation")
            ax_arr[1].axvline(train_n_iter, linestyle='--', color='black')

            for c_len in c_len_arr:
                e_mean_resp = ch_results['c_len_{}_mean_e_resp'.format(c_len)]
                e_std_resp = ch_results['c_len_{}_std_e_resp'.format(c_len)]
                i_mean_resp = ch_results['c_len_{}_mean_i_resp'.format(c_len)]
                i_std_resp = ch_results['c_len_{}_std_i_resp'.format(c_len)]
                timesteps = np.arange(0, len(e_mean_resp))

                color = next(ax_arr[0]._get_lines.prop_cycler)['color']

                ax_arr[0].plot(timesteps, e_mean_resp, label='clen_{}'.format(c_len), color=color)
                ax_arr[0].fill_between(
                    timesteps,
                    e_mean_resp + e_std_resp,
                    e_mean_resp - e_std_resp,
                    alpha=0.2, color=color)

                ax_arr[1].plot(timesteps, i_mean_resp, label='clen_{}'.format(c_len), color=color)
                ax_arr[1].fill_between(
                    timesteps,
                    i_mean_resp + i_std_resp,
                    i_mean_resp - i_std_resp,
                    alpha=0.2, color=color)

            ax_arr[0].legend()
            f_resp.savefig(os.path.join(per_chan_r_dir, 'resp_ch_{}.jpg'.format(ch_idx)), format='jpg')

            plt.close(f_iou)
            plt.close(f_resp)


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    random_seed = 10

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/contour_dataset_multiple_runs/' \
        'positive_lateral_weights_with_independent_BN_best_gain_curves/' \
        'run_1/best_accuracy.pth'

    # Immutable Initialization
    # ========================
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    start_time = datetime.now()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))

    results_dir = os.path.dirname(saved_model)
    results_dir = os.path.join(results_dir, 'dynamic_responses')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    prev_stored_optimal_stimuli_file = os.path.join(results_dir, 'optimal_stimuli.pickle')
    optimal_stimuli_dict = None
    if os.path.exists(prev_stored_optimal_stimuli_file):
        with open(prev_stored_optimal_stimuli_file, 'rb') as handle:
            optimal_stimuli_dict = pickle.load(handle)

    # Main function
    # =============
    main(net, optimal_stimuli_dict, results_dir)

    # ===================================================================================
    # End
    # ===================================================================================
    print("Running script took {}".format(datetime.now() - start_time))

    import pdb
    pdb.set_trace()
