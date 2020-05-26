# ---------------------------------------------------------------------------------------
# Get  or view predictions of a trained model on the contour integration dataset
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
    random_seed = 5
    data_dir = "./data/channel_wise_optimal_full14_frag7"

    save_predictions = True

    # # Control Model
    # ----------------
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    cont_int_layer = new_control_models.ControlMatchParametersLayer(
        lateral_e_size=15, lateral_i_size=15)
    saved_model = \
        'results/new_model_resnet_based/' \
        'ContourIntegrationResnet50_ControlMatchParametersLayer_20200508_223114_baseline' \
        '/best_accuracy.pth'


    # # Model
    # # -----
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # saved_model =\
    #     "./results/new_model_resnet_based/" \
    #     "ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20200508_222333_baseline" \
    #     "/best_accuracy.pth"

    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

    # Immutable
    # ---------------------------------------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.load_state_dict(torch.load(saved_model, map_location=device))

    net.edge_extract.register_forward_hook(edge_extract_cb)
    net.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
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
        c_len_arr=None,
        beta_arr=None,
        alpha_arr=None,
        gabor_set_arr=None
    )

    val_data_loader = DataLoader(
        dataset=val_data_set,
        num_workers=4,
        batch_size=1,
        shuffle=True,
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
    net.eval()
    e_loss = 0
    e_iou = 0

    # Where to save predictions
    if save_predictions:
        list_of_files = val_data_loader.dataset.labels

        model_results_dir = os.path.dirname(saved_model)
        preds_dir = os.path.join(model_results_dir, 'predictions')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)

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

                if save_predictions:
                    # plt.imsave(
                    #     fname=os.path.join(preds_dir, list_of_files[iteration].split('/')[-1]),
                    #     arr=np.squeeze(label_out.detach().cpu().numpy()),
                    #     cmap=plt.cm.gray,
                    # )
                    np.save(file=os.path.join(preds_dir, list_of_files[iteration].split('/')[-1]), arr=np.squeeze(label_out.detach().cpu().numpy()))

        e_loss = e_loss / len(val_data_loader)
        e_iou = e_iou / len(val_data_loader)
        print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
