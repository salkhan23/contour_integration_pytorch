# ---------------------------------------------------------------------------------------
#  Visualize First layer kernels of a Model. Find Gabor Fit parameters for the kernels
# ---------------------------------------------------------------------------------------
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models import alexnet

# This file uses modules from the parent directory
# to import this, the parent directory needs to be added to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gabor_fits


if __name__ == "__main__":
    net = alexnet(pretrained=True)

    random_seed = 7

    # -------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # -----------------------------------------------------------------------------------
    # Plot the kernels
    # -----------------------------------------------------------------------------------
    l1_kernels = net.features[0].weight
    l1_kernels = l1_kernels.cpu().detach().numpy()
    out_ch, in_ch, r, c = l1_kernels.shape

    # Tile image dimensions
    margin = 1

    cols = out_ch * r + (out_ch - 1) * margin
    rows = in_ch * c + (in_ch - 1) * margin

    tiled_image_filters = np.zeros((rows, cols))
    tiled_image_fitted_gabors = np.zeros_like(tiled_image_filters)

    for k_idx in range(out_ch):

        print("Processing kernel {}".format(k_idx))

        # 1. Find best fit Gabor Params
        kernel = np.transpose(l1_kernels[k_idx, ], axes=(1, 2, 0))
        frag_size = np.array(kernel.shape[0:2])

        for ch_idx in range(in_ch):

            normed_kernel = kernel[:, :, ch_idx]
            normed_kernel = (normed_kernel - normed_kernel.min()) / (normed_kernel.max() - normed_kernel.min())

            tiled_image_filters[
               ch_idx * (c + margin):  ch_idx * (c + margin) + c,
               k_idx * (r + margin):  k_idx * (r + margin) + r,
            ] = normed_kernel

        # best fit returns a list of 3 Gabor fits: one for each channel
        # Each gabor fit is itself a list with the following elements:
        # [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma]
        best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(kernel, verbose=1)

        # -----------------------------------------------------------------------------
        # Generate Single channel Gabors for each channel seperately
        # -----------------------------------------------------------------------------
        for item_idx, item in enumerate(best_fit_params_list):
            if item is not None:
                params = [{
                    'x0': item[0],
                    'y0': item[1],
                    'theta_deg': item[2],
                    'amp': item[3],
                    'sigma': item[4],
                    'lambda1': item[5],
                    'psi': item[6],
                    'gamma': item[7]
                }]

                frag = gabor_fits.get_gabor_fragment(params, frag_size)

                tiled_image_fitted_gabors[
                    item_idx * (c + margin):  item_idx * (c + margin) + c,
                    k_idx * (r + margin):  k_idx * (r + margin) + r,
                ] = frag[:, :, 0]  # All Gabors are the same

        # ----------------------------------------------------------------------
        # Generate Single Gabor with Three channels
        # ----------------------------------------------------------------------
        # # If a fit for any channel is not found, None is returned
        # valid_best_fits = [item for item in best_fit_params_list if item is not None]
        #
        # if valid_best_fits:
        #
        #     valid_best_fits = np.array(valid_best_fits)
        #
        #     # 2. Create a Gabor fragment from the found params
        #     if valid_best_fits.shape[0] == 3:
        #         params = [
        #             {
        #                 'x0': valid_best_fits[0, 0],
        #                 'y0': valid_best_fits[0, 1],
        #                 'theta_deg': valid_best_fits[0, 2],
        #                 'amp': valid_best_fits[0, 3],
        #                 'sigma': valid_best_fits[0, 4],
        #                 'lambda1': valid_best_fits[0, 5],
        #                 'psi': valid_best_fits[0, 6],
        #                 'gamma': valid_best_fits[0, 7]
        #             },
        #             {
        #                 'x0': valid_best_fits[1, 0],
        #                 'y0': valid_best_fits[1, 1],
        #                 'theta_deg': valid_best_fits[1, 2],
        #                 'amp': valid_best_fits[1, 3],
        #                 'sigma': valid_best_fits[1, 4],
        #                 'lambda1': valid_best_fits[1, 5],
        #                 'psi': valid_best_fits[1, 6],
        #                 'gamma': valid_best_fits[1, 7]
        #             },
        #             {
        #                 'x0': valid_best_fits[2, 0],
        #                 'y0': valid_best_fits[2, 1],
        #                 'theta_deg': valid_best_fits[2, 2],
        #                 'amp': valid_best_fits[2, 3],
        #                 'sigma': valid_best_fits[2, 4],
        #                 'lambda1': valid_best_fits[2, 5],
        #                 'psi': valid_best_fits[2, 6],
        #                 'gamma': valid_best_fits[2, 7]
        #             },
        #         ]
        #
        #     else:
        #         # Just use the gabor channel with the highest amplitude
        #         amps_arr = valid_best_fits[:, 3]
        #         ch_idx = np.argmax(amps_arr)
        #
        #         # # print(valid_best_fits)
        #         # print("Gabor parameters with highest amp: chan {}, value {}".format(
        #         #     ch_idx, valid_best_fits[ch_idx, 3]))
        #
        #         params = [{
        #             'x0': valid_best_fits[ch_idx, 0],
        #             'y0': valid_best_fits[ch_idx, 1],
        #             'theta_deg': valid_best_fits[ch_idx, 2],
        #             'amp': valid_best_fits[ch_idx, 3],
        #             'sigma': valid_best_fits[ch_idx, 4],
        #             'lambda1': valid_best_fits[ch_idx, 5],
        #             'psi': valid_best_fits[ch_idx, 6],
        #             'gamma': valid_best_fits[ch_idx, 7]
        #         }]
        #
        #     frag = gabor_fits.get_gabor_fragment(params, frag_size)
        #
        #     # Display frag and generated Gabor
        #     f, ax_arr = plt.subplots(1, 2)
        #     display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
        #     ax_arr[0].imshow(display_kernel)
        #     ax_arr[0].set_title('kernel')
        #     ax_arr[1].imshow(frag)
        #     ax_arr[1].set_title('fragment')
        #
        #     import pdb
        #     pdb.set_trace()

    plt.figure(figsize=(15, 3))
    plt.imshow(tiled_image_filters)
    plt.title("Kernels")

    plt.figure(figsize=(15, 3))
    plt.imshow(tiled_image_fitted_gabors)
    plt.title("Fitted Gabors")

    # Get Larger scrollable images in your browser. Need to pip install mpld3
    import mpld3
    mpld3.show()

    import pdb
    pdb.set_trace()
