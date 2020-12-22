# ---------------------------------------------------------------------------------------
#  Debug: For a trained model, plot histograms of excitatory and inhibitory activations
#  values over individual recurrent steps
#
#  Can also be used to view the Excitatory and inhibitory activations of the model for
#  a single image
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import pickle
import dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import models.new_piech_models as new_piech_models


def display_image(input_img, label1, label_out1):
    label_out1 = np.squeeze(label_out1)
    label1 = np.squeeze(label1)
    display_img = np.squeeze(input_img)
    display_img = np.transpose(display_img, axes=(1, 2, 0))

    plt.figure()
    plt.imshow(label_out1)
    plt.title("Prediction")

    plt.figure()
    plt.imshow(label1)
    plt.title("Label")

    plt.figure()
    plt.imshow(display_img)
    plt.title("Input")


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------------------------
    random_seed = 10
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    data_dir = "./data/channel_wise_optimal_full14_frag7"

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, store_recurrent_acts=True)

    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)
    saved_model = \
        './results/positive_weights' \
        '/ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20201220_194703_baseline' \
        '/best_accuracy.pth'

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=dev))

    device = dev

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    print(">>>> Setting up data loaders")

    metadata_file = os.path.join(data_dir, 'dataset_metadata.pickle')
    with open(metadata_file, 'rb') as h:
        metadata = pickle.load(h)

    # Pre-processing
    pre_process_transforms = transforms.Compose([
        transforms.Normalize(mean=metadata['channel_mean'], std=metadata['channel_std']),
    ])

    val_data_set = dataset.Fields1993(
        data_dir=os.path.join(data_dir, 'val'),
        bg_tile_size=metadata["full_tile_size"],
        transform=pre_process_transforms,
        subset_size=None,
        c_len_arr=[7, 9],
        beta_arr=None,
        alpha_arr=[0],
        gabor_set_arr=None
    )

    val_data_loader = DataLoader(
        dataset=val_data_set,
        num_workers=4,
        batch_size=1,
        shuffle=False,  # Do not change needed to save predictions with correct file names
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    print(">>>> Starting Main Loop")

    net.eval()
    n_iters = net.contour_integration_layer.n_iters

    # Collect activations over all images
    x_activations_per_iter = {}
    y_activations_per_iter = {}
    for iter_idx in range(n_iters):
        x_activations_per_iter[iter_idx] = np.array([])
        y_activations_per_iter[iter_idx] = np.array([])

    with torch.no_grad():
        for iteration, (img, label) in enumerate(val_data_loader, 0):

            img = img.to(device)
            label = label.to(device)
            label_out = net(img)

            # display_image(img, label, label_out)

            # f, ax_arr = plt.subplots(2, n_iters, figsize=(15, 5))

            for iter_idx in range(n_iters):
                excite_act = net.contour_integration_layer.x_per_iteration[iter_idx].detach().cpu().numpy()
                excite_act = np.squeeze(excite_act)
                inhibit_act = net.contour_integration_layer.y_per_iteration[iter_idx].detach().cpu().numpy()
                inhibit_act = np.squeeze(inhibit_act)

                # # Plot the activations
                # ax_arr[0, iter_idx].imshow(excite_act.sum(axis=0))
                # ax_arr[1, iter_idx].imshow(inhibit_act.sum(axis=0))

                x_activations_per_iter[iter_idx] = \
                    np.append(x_activations_per_iter[iter_idx], excite_act.flatten())

                y_activations_per_iter[iter_idx] = \
                    np.append(y_activations_per_iter[iter_idx], inhibit_act.flatten())

            if iteration == 100:
                break

        # Plot Histograms of activations (non-rectified)
        f_x, x_ax_arr = plt.subplots(1, n_iters, figsize=(15, 5))
        f_y, y_ax_arr = plt.subplots(1, n_iters, figsize=(15, 5))

        for iter_idx in range(n_iters):
            x_ax_arr[iter_idx].hist(x_activations_per_iter[iter_idx], bins=20)
            y_ax_arr[iter_idx].hist(y_activations_per_iter[iter_idx], bins=20)

        f_x.suptitle("X Activations per recurrent step")
        f_y.suptitle("Y Activations per recurrent step")

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
