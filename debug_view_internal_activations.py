# ---------------------------------------------------------------------------------------
# View Internal inhibitory/excitatory activations of a trained contour integration Model
#
# NOTE: The following temp code needs to be added to models/new_piech_models.py
#
#   >> # # Debug
#   >> # idx = ff.argmax()  # This is the index in the flattened array
#   import matplotlib.pyplot as plt
#   import numpy as np
#   f, ax_arr = plt.subplots(2, 5, figsize=(15, 5))
#   ax_arr[0, 0].set_ylabel("f(x)")
#   ax_arr[0, 1].set_ylabel("f(y)")
#   >> ...
#
#   >> for i in range(self.n_iters):
#   >> ..
#   >> f_x = nn.functional.relu(x)
#   >> f_y = nn.functional.relu(y)
#   disp_x = f_x.detach().cpu().numpy()
#   disp_x = np.squeeze(disp_x)
#   disp_x = np.sum(disp_x, axis=0)  # some across channels
#
#   disp_y = f_y.detach().cpu().numpy()
#   disp_y = np.squeeze(disp_y)
#   disp_y = np.sum(disp_y, axis=0)  # some across channels
#
#   im_x = ax_arr[0, i].imshow(disp_x)
#   plt.colorbar(im_x, ax=ax_arr[0, i], orientation='horizontal')
#   im_y = ax_arr[1, i].imshow(disp_y)
#   plt.colorbar(im_y, ax=ax_arr[1, i], orientation='horizontal')
# ---------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import utils

edge_extract_act = []
cont_int_in_act = []
cont_int_out_act = []


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at Edge Extract layer
        Callback to retrieve the activations output of Edge Extract layer
    """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer
        Callback to retrieve the input & output activations of the contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    random_seed = 5
    data_dir = "./data/channel_wise_optimal_full14_frag7"

    save_predictions = True

    # Model
    # -----
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, b=0.847)
    # saved_model =\
    #     "./results/new_model_resnet_based/" \
    #     "ContourIntegrationResnet50_CurrentSubtractInhibitLayer_run_1_20200924_183734" \
    #     "/best_accuracy.pth"
    saved_model = \
        '/home/salman/Desktop/' \
        'ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20201208_093455_postive weights_clip_every_timestep_150_epochs/' \
        'best_accuracy.pth'

    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.load_state_dict(torch.load(saved_model, map_location=device))

    print("Model Loaded")

    # -----------------------------------------------------------------------------------
    # Setup the data loader
    print("Setting up data loaders")

    metadata_file = os.path.join(data_dir, 'dataset_metadata.pickle')
    with open(metadata_file, 'rb') as h:
        metadata = pickle.load(h)

    # Pre-processing
    pre_process_transforms = transforms.Compose([
        transforms.Normalize(mean=metadata['channel_mean'], std=metadata['channel_std']),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ])

    val_data_set = dataset.Fields1993(
        data_dir=os.path.join(data_dir, 'val'),
        bg_tile_size=metadata["full_tile_size"],
        transform=pre_process_transforms,
        subset_size=None,
        c_len_arr=[9],
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
    # Loss Function
    # -----------------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss().to(device)
    detect_thres = 0.5

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("Starting Main Loop")
    net.eval()
    e_loss = 0
    e_iou = 0

    # Where to save predictions
    if save_predictions:
        list_of_files = val_data_loader.dataset.labels

        with torch.no_grad():
            for iteration, (img, label) in enumerate(val_data_loader, 0):
                img = img.to(device)
                label = label.to(device)

                label_out = net(img)
                batch_loss = criterion(label_out, label.float())

                preds = (torch.sigmoid(label_out) > detect_thres)
                iou = utils.intersection_over_union(
                    preds.float(), label.float()).cpu().detach().numpy()
                e_iou += iou

                # Before visualizing Sigmoid the output. This is already done in the loss function
                label_out = torch.sigmoid(label_out)

                # Display Image Prediction and Labels
                # --------------------------------------------------------------------
                label_out = np.squeeze(label_out)
                plt.figure()
                plt.imshow(label_out)
                plt.title("Prediction")

                label = np.squeeze(label)
                plt.figure()
                plt.imshow(label)
                plt.title("Label")

                display_img = np.squeeze(img)
                display_img = np.transpose(display_img, axes=(1, 2, 0))
                plt.imshow(display_img)
                plt.title("Input")

                import pdb
                pdb.set_trace()

    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
