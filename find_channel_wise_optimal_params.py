# ---------------------------------------------------------------------------------------
#  Starting with a fix set of base gabor params. Find the params
#  the result in the highest response. Each base param set is tunable for
#    (1) Orientation.
#
#   Each Base Gabor Set generates fragments that look like line segments.
#
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

import torch
from torchvision.models import alexnet

import gabor_fits
import fields1993_stimuli


edge_extract_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at edge_extract layer """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def get_center_neuron_responses(model, in_img):
    """ TODO: Device should be passed in or implicitly picked up """
    # Image pre-processing
    # --------------------
    # [0, 1] Pixel Range
    test_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())
    # Normalize using ImageNet mean/std
    model_in_img = (test_img - np.array([[0.4208942, 0.4208942, 0.4208942]])) / \
        np.array([0.15286704, 0.15286704, 0.15286704])

    model_in_img = torch.from_numpy(np.transpose(model_in_img, axes=(2, 0, 1)))
    model_in_img = model_in_img.to(device)
    model_in_img = model_in_img.float()
    model_in_img = model_in_img.unsqueeze(dim=0)

    # Get Model activation
    global edge_extract_act
    edge_extract_act = 0
    _ = model(model_in_img)

    center_neuron_extract_out = \
        edge_extract_act[0, :, edge_extract_act.shape[2] // 2, edge_extract_act.shape[3] // 2]

    return center_neuron_extract_out


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    frag_size = np.array([7, 7])

    image_size = np.array([256, 256, 3])
    image_center = image_size[0:2] // 2

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Base Gabors
    # -----------------------------------------------------------------------------------
    gabor_parameters_list = [
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

    bg_list = [0, 0, 0, 0, 255, None, 0, 255, 0, 255]

    gabor_parameters = []
    for set_idx, gabor_set in enumerate(gabor_parameters_list):
        params = gabor_fits.convert_gabor_params_list_to_dict(gabor_set)

        for chan_params in params:
            chan_params['bg'] = bg_list[set_idx]

        gabor_parameters.append(params)

    # Debug - Plot all Gabor Fragments
    fig = plt.figure(figsize=(10, 5))
    for p_idx, params in enumerate(gabor_parameters):
        fig.add_subplot(2, 5, p_idx + 1)
        plt.imshow(gabor_fits.get_gabor_fragment(params, frag_size))

    # import pdb
    # pdb.set_trace()

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("===> Loading Model")
    net = alexnet(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    net.features[0].register_forward_hook(edge_extract_cb)

    n_channels = 64
    # -----------------------------------------------------------------------------------
    # Find Optimal Stimuli
    # -----------------------------------------------------------------------------------
    # Tunable Values
    orient_arr = np.arange(0, 180, 5)

    list_of_optimal_stimuli = []

    for k_idx in range(n_channels):
        print("{0} kernel {1} {0}".format('*'*40, k_idx))

        tgt_neuron_acts = np.zeros((len(gabor_parameters), len(orient_arr)))  # [Gabor params, orientations]
        tgt_neuron_max_act = 0
        kernel_optimal_params = None

        for base_gp_idx, base_gabor_params in enumerate(gabor_parameters):
            print("Processing Base Gabor Param Set {}".format(base_gp_idx))

            for o_idx, orient in enumerate(orient_arr):

                # Change Orientation
                # ------------------
                gabor_params = copy.deepcopy(base_gabor_params)

                for ch_idx in range(len(gabor_params)):
                    gabor_params[ch_idx]["theta_deg"] = orient

                # Create Test Image
                # -----------------
                # Create a fragment from gabor_params, place in center of image.
                # This location is optimized to fit within the receptive fields
                # of centrally located neurons. Next, get target neuron responses.

                frag = gabor_fits.get_gabor_fragment(gabor_params, spatial_size=frag_size)

                bg = base_gabor_params[0]['bg']
                if bg is None:
                    bg = fields1993_stimuli.get_mean_pixel_value_at_boundary(frag)
                test_image = np.ones(image_size, dtype='uint8') * bg

                test_image[
                    image_center[0] - frag_size[0] // 2: image_center[0] + frag_size[0] // 2 + 1,
                    image_center[0] - frag_size[0] // 2: image_center[0] + frag_size[0] // 2 + 1,
                    :,
                ] = frag

                # Debug - Show Test Image
                # plt.figure()
                # plt.imshow(test_image)
                # plt.title("Input Image")

                # Get Center Neuron Responses
                # ---------------------------
                center_neuron_responses = get_center_neuron_responses(net, test_image)

                tgt_neuron_act = center_neuron_responses[k_idx]
                tgt_neuron_acts[base_gp_idx, o_idx] = tgt_neuron_act

                # Save Neuron Responses
                tgt_neuron_act = center_neuron_responses[k_idx]
                tgt_neuron_acts[base_gp_idx, o_idx] = tgt_neuron_act

                # check for optimal Stimulus
                if tgt_neuron_act > tgt_neuron_max_act:
                    tgt_neuron_max_act = tgt_neuron_act

                    kernel_optimal_params = copy.deepcopy(gabor_params)

                    # Attach some additional Meta Data
                    for ch_idx in range(len(gabor_params)):
                        kernel_optimal_params[ch_idx]['optimal_stimulus_act'] = tgt_neuron_max_act
                        kernel_optimal_params[ch_idx]['base_gabor_set'] = base_gp_idx

                        max_active_neuron = np.argmax(center_neuron_responses)
                        kernel_optimal_params[ch_idx]['is_max_active'] = (max_active_neuron == k_idx)
                        kernel_optimal_params[ch_idx]['max_active_act'] = \
                            center_neuron_responses[max_active_neuron]

                # Debug - Display All channel responses to individual test image
                # plt.figure()
                # plt.plot(center_neuron_responses)
                # plt.title("Center Neuron Activations. Base Gabor Set {}. Orientation {}".format(
                #     base_gp_idx, orient))

            # # -----------------------------------------
            # # Debug - Tuning Curve for Individual base Gabor Params
            # plt.figure()
            # plt.plot(orient_arr, tgt_neuron_acts[base_gp_idx, :])
            # plt.title("Neuron {}: responses vs Orientation. Gabor Set {}".format(k_idx, base_gp_idx))
            # import pdb
            # pdb.set_trace()

        # # ---------------------------------------
        # Save optimal tuning curve
        # Note stored only for first channel
        optimal_base_gabor_set = kernel_optimal_params[0]['base_gabor_set']
        kernel_optimal_params[0]['orient_arr'] = orient_arr
        kernel_optimal_params[0]['orient_tuning_curve'] = tgt_neuron_acts[optimal_base_gabor_set,]

        # TODO: Mark Good Neurons: (1) Amp can be check using optimal_stimulus_act
        # TODO: How to check for clear orientation tuning preference
        # Li2006 - " Neurons that were not responsive to single bars or did not
        # show a clear orientation tuning preference were skipped."

        # Store the optimal params
        list_of_optimal_stimuli.append(kernel_optimal_params)

        # # Debug - 1: plot tuning curves for all gabor sets
        # plt.figure()
        # for base_gp_idx, base_gabor_params in enumerate(gabor_parameters):
        #
        #     if base_gp_idx == kernel_optimal_params[0]['base_gabor_set']:
        #         line_width = 5
        #     else:
        #         line_width = 2
        #
        #     plt.plot(orient_arr, tgt_neuron_acts[base_gp_idx, ],
        #              label='param set {}'.format(base_gp_idx), linewidth=line_width)
        #
        # plt.legend()
        # plt.title("Kernel {}. Max Active Base Set {}. Is most Responsive to this stimulus {}".format(
        #     k_idx, kernel_optimal_params[0]['base_gabor_set'], kernel_optimal_params[0]['is_max_active']))
        #
        # import pdb
        # pdb.set_trace()

        # # Debug 2: plot tuning curve for optimal gabor set
        # plt.figure()
        # plt.title(
        #     "Kernel {}: Tuning Curve.\n Base Gabor Param set {}. Is Max Responsive Neuron {}. "
        #     "Max Activation {:0.2f}".format(
        #         k_idx,
        #         kernel_optimal_params[0]['base_gabor_set'],
        #         kernel_optimal_params[0]['is_max_active'],
        #         kernel_optimal_params[0]['optimal_stimulus_act']))
        #
        # plt.plot(orient_arr, kernel_optimal_params[0]['orient_tuning_curve'])
        #
        # import pdb
        # pdb.set_trace()

    # --------------------------------------------------
    # Save Pickle File
    params_pickle_file = 'channel_wise_optimal_stimuli.pickle'
    with open(params_pickle_file, 'wb') as handle:
        pickle.dump(list_of_optimal_stimuli, handle)

    # Some Statistical Details

    # Distribution of base gabor sets used
    preferred_base_gabors = [item[0]['base_gabor_set'] for item in list_of_optimal_stimuli]
    unique, counts = np.unique(preferred_base_gabors, return_counts=True)
    plt.figure()
    plt.plot(unique, counts)
    plt.title("Histogram Base Gabor Sets")
    plt.xlabel("Gabor Set")
    plt.ylabel("Counts")
    print("Base Gabor Sets counts {}".format(counts))

    # How many neurons are most active fore their optimal stimulus
    is_max_active_list = [item[0]['is_max_active'] for item in list_of_optimal_stimuli]
    print("{} Neurons are most active for their optimal stimulus".format(np.array(is_max_active_list).sum()))

    # How many Neurons have activation above a threshold
    preferred_stimuli_acts = np.array([item[0]['optimal_stimulus_act'] for item in list_of_optimal_stimuli])

    plt.figure()
    plt.plot(preferred_stimuli_acts)
    plt.plot(
        np.nonzero(is_max_active_list)[0],
        preferred_stimuli_acts[np.nonzero(is_max_active_list)[0]],
        linestyle='None', marker='x', color='red', label='most responsive')
    plt.title("Responses to Optimal Stimuli")
    plt.xlabel("Channel")
    plt.ylabel("Activation")
    plt.legend()

    # All Activations above threshold
    threshold = 15
    is_above_threshold = preferred_stimuli_acts > threshold
    print("{} neurons have activation above {}".format(is_above_threshold.sum(), threshold))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()

