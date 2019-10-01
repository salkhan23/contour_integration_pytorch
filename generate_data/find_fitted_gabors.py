# ---------------------------------------------------------------------------------------
#  For each edge extracting kernel
#  1. Find the best fit Gabors
#  2. Generate Gabor fragment from found parameters
#  3. Verify that the target neuron is the one most responding the to generated fragment
#  4. Store the generated Gabor Parameters
#
#  Additionally certain restrictions are added to the found gabor parameters to make
#  the test images easier to construct.
#       1. Spatial extend is limited to fit within a 11x11 tile so that the fragment
#          can be blended into the background easily
#       2. A single orientation is used. Currently the stimuli generating script uses a
#          single fragment orientation.
#       3. The x0,y0 location is forced to be (0,0)
#
#  Found parameters are saved in fill, fitted Gabors. To generate pickle files expected
#  by the generation script see misc/create Gabor_params_file
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models import alexnet

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gabor_fits
import fields1993_stimuli

edge_extract_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    write_file = 'fitted_gabors_params.txt'

    fragment_size = np.array([11, 11])
    full_tile_size = np.array([32, 32])

    image_size = np.array([512, 512, 3])

    # Immutable
    # --------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    net = alexnet(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    net.features[0].register_forward_hook(edge_extract_cb)

    l1_kernels = net.features[0].weight  # Edge extracting layer of alexnet
    l1_kernels = l1_kernels.cpu().detach().numpy()
    out_ch, in_ch, r, c = l1_kernels.shape

    matched_gabors = 0
    file_handle = open(write_file, 'w+')
    file_handle.write("[x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma]\n")

    for k_idx in range(out_ch):

        print("Processing kernel {}".format(k_idx))
        kernel = np.transpose(l1_kernels[k_idx, ], axes=[1, 2, 0])

        # Best fit returns a list of 3 Gabor fits: one for each channel.
        # Each gabor fit is itself a list with the following elements:
        # [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma]
        best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(kernel, verbose=0)
        valid_best_fits = [item for item in best_fit_params_list if item is not None]

        if len(valid_best_fits) == 3:

            valid_best_fits = np.array(valid_best_fits)

            params = []

            # Single Orientation. Use the orientation of the channel with the highest amplitude
            amps = valid_best_fits[:, 3]
            max_amps_idx = np.argmax(np.abs(amps))

            single_theta = valid_best_fits[max_amps_idx, 2]

            for ch_idx, ch_params in enumerate(valid_best_fits):
                params.append(
                    {
                        'x0': 0,
                        'y0': 0,
                        'theta_deg': single_theta,
                        'amp': ch_params[3],
                        'sigma': np.min((ch_params[4], 2)),
                        'lambda1': ch_params[5],
                        'psi': ch_params[6],
                        'gamma': ch_params[7]
                    }
                )

            frag = gabor_fits.get_gabor_fragment(params, fragment_size)

            # # Display frag and generated Gabor
            # f, ax_arr = plt.subplots(1, 2)
            # display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            # ax_arr[0].imshow(display_kernel)
            # ax_arr[0].set_title('kernel')
            # ax_arr[1].imshow(frag)
            # ax_arr[1].set_title('fragment')
            #
            # import pdb
            # pdb.set_trace()

            # Generate a Test image with Fragment in the center
            # -------------------------------------------------
            center_tile_start = image_size[0:2] // 2 - fragment_size[0:2] // 2

            test_image = np.zeros(image_size, dtype=np.uint8)

            c_len = 1
            test_image, path_fragment_starts = fields1993_stimuli.add_contour_path_constant_separation(
                img=test_image,
                frag=frag,
                frag_params=params,
                c_len=c_len,
                beta=0,
                alpha=0,
                d=full_tile_size[0],
                rand_inter_frag_direction_change=False,
                base_contour='sigmoid'
            )

            # plt.figure()
            # plt.imshow(test_image)
            # plt.title("Test image. Contour only, Zero Bg. c_len = {}".format(c_len))

            # Setup image for input to model
            test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
            input_image = (test_image - np.array([[0.4208942, 0.4208942, 0.4208942]])) / \
                np.array([0.15286704, 0.15286704, 0.15286704])  # image pre-processing 0 mean,1 STD

            input_image = torch.from_numpy(np.transpose(input_image, axes=(2, 0, 1)))
            input_image = input_image.to(device)
            input_image = input_image.float()
            input_image = input_image.unsqueeze(dim=0)

            edge_extract_act = 0
            label_out = net(input_image)

            center_neuron_extract_out = edge_extract_act[0, :, 127 // 2, 127 // 2]

            max_active_neuron = np.argmax(center_neuron_extract_out)
            string = "Target Neuron: {}. Max Active Neuron {}. is Match = {}".format(
                 k_idx, max_active_neuron, k_idx == max_active_neuron)
            print(string)

            # Plot Center Neuron Activations
            plt.figure()
            plt.plot(center_neuron_extract_out)
            plt.title(string)

            if k_idx == max_active_neuron:
                matched_gabors += 1
                print("Okay Neurons Count {}".format(matched_gabors))

                # Plot Center Neuron Activations
                plt.figure()
                plt.plot(center_neuron_extract_out)
                plt.title(string)

                # Plot frag and generated Gabor
                f, ax_arr = plt.subplots(1, 2)
                display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
                ax_arr[0].imshow(display_kernel)
                ax_arr[0].set_title('kernel {}'.format(k_idx))
                ax_arr[1].imshow(frag)
                ax_arr[1].set_title('fragment')

                # write to file
                file_handle.write("Kernel {}. Amplitude {:0.2f}\n".format(
                    k_idx, center_neuron_extract_out[max_active_neuron]))

                # print params
                for item_idx, item in enumerate(params):

                    print("{}: (x0,y0) ({:0.2f}, {:0.2f}), theta {:0.2f}, amp {:0.2f}, sigma {:0.2f}, "
                          "lambda {:0.2f}, psi {:0.2f}, gamma {:0.2f}".format(
                            item_idx,
                            item['x0'],
                            item['y0'],
                            item['theta_deg'],
                            item['amp'],
                            item['sigma'],
                            item['lambda1'],
                            item['psi'],
                            item['gamma']))

                    # Format [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma]
                    file_handle.write(
                        "[{:0.2f}, {:0.2f}, {:0.2f},{:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}],\n".format(
                            item['x0'],
                            item['y0'],
                            item['theta_deg'],
                            item['amp'],
                            item['sigma'],
                            item['lambda1'],
                            item['psi'],
                            item['gamma'],
                    ))

            import pdb
            pdb.set_trace()

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("Final Count of good neurons {}".format(matched_gabors))

    file_handle.write("Number of fitted kernels {}\n".format(matched_gabors))
    file_handle.close()

    import pdb
    pdb.set_trace()
