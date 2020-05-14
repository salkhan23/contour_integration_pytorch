# ---------------------------------------------------------------------------------------
# Get performance of a Trained Edge Extraction Model over the Validation  Dataset
# Also visualize model output and contour integration layer activations
# ---------------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset_edge
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


def zero_mean_and_unit_var(in_img):
    mean = np.mean(in_img)
    var = np.var(in_img)

    out_img = (in_img - mean) / var

    return out_img


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 5
    data_set_dir = "./data/edge_detection_data_set"

    # # Control Model
    # ----------------
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # # cont_int_layer = new_control_models.ControlMatchParametersLayer(
    # #     lateral_e_size=15, lateral_i_size=15)
    # saved_model = \
    #     'results/biped' \
    #     '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base' \
    #     '/last_epoch.pth'

    # Model
    # -----
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    saved_model = \
        'results/biped' \
        '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base' \
        '/last_epoch.pth'

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

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
    # Data loader
    # -----------------------------------------------------------------------------------
    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    pre_process_transforms = transforms.Compose([
        transforms.Normalize(mean=ch_mean, std=ch_std),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0),
    ])

    val_set = dataset_edge.EdgeDataSet(
        data_dir=os.path.join(data_set_dir, 'val'),
        transform=pre_process_transforms
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Loss Function
    # -----------------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss().to(device)
    detect_thres = 0.3

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    net.eval()
    e_loss = 0
    e_iou = 0

    with torch.no_grad():
        for iteration, (img, label) in enumerate(val_data_loader, 1):
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

            #  Plot input image, label and prediction
            # ---------------------------------------------------------------------------
            display_img = img.detach().cpu().numpy()
            display_img = np.squeeze(display_img)
            display_img = np.transpose(display_img, axes=(1, 2, 0))
            display_img = (display_img - display_img.min()) / \
                          (display_img.max() - display_img.min())

            f, ax_arr = plt.subplots(2, 2, figsize=(11,9))

            ax_arr[0][0].imshow(display_img)
            ax_arr[0][0].set_title("Image")

            label = label.detach().cpu().numpy()
            label = np.squeeze(label)

            p = ax_arr[1][0].imshow(label)
            # f.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            ax_arr[1][0].set_title("GT")

            label_out = label_out.detach().cpu().numpy()
            label_out = np.squeeze(label_out)

            p = ax_arr[0][1].imshow(label_out)
            f.colorbar(p, ax=ax_arr[0][1], orientation="vertical")
            ax_arr[0][1].set_title('Model Output')

            p1 = ax_arr[1][1].imshow(np.squeeze(preds))
            # f.colorbar(p, ax=ax_arr[1][1], orientation="horizontal")
            ax_arr[1][1].set_title('Threshold {} output. IoU ={:0.4f}'.format(detect_thres, iou))

            # Plot Contour Integration Layer Input and output
            # ---------------------------------------------------------------------------
            # sum over all channels
            edge_out = edge_extract_act.squeeze().max(axis=0)
            cont_int_in = cont_int_in_act.squeeze().max(axis=0)
            cont_int_out = cont_int_out_act.squeeze().max(axis=0)

            f2, ax_arr = plt.subplots(1, 3)
            p = ax_arr[0].imshow(edge_out)
            f2.colorbar(p, ax=ax_arr[0], orientation="horizontal")
            ax_arr[0].set_title('Edge Extraction Out')

            p = ax_arr[1].imshow(cont_int_in)
            f2.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            ax_arr[1].set_title('Contour Integration In')

            p = ax_arr[2].imshow(cont_int_out)
            f2.colorbar(p, ax=ax_arr[2], orientation="horizontal")
            ax_arr[2].set_title('Contour Integration Out')

            # visualize the difference added by the contour integration layer
            # ----------------------------------------------------------------
            # There seems to be a scale difference between the input and output of the
            # contour integration layer normalize the activations before looking
            # at differences

            # This way seems to give the best difference
            # TODO: what is the justification for this way
            diff = cont_int_out - cont_int_in
            gain = cont_int_out / (cont_int_in + 0.0001)

            normed_diff = zero_mean_and_unit_var(diff)
            normed_gain = zero_mean_and_unit_var(gain)

            f3, ax_arr = plt.subplots(1, 2)
            f3.suptitle("Modifications added by the contour integration layer")
            p = ax_arr[0].imshow(
                normed_diff, cmap='seismic', vmin=-normed_diff.max(), vmax=normed_diff.max())
            f.colorbar(p, ax=ax_arr[0], orientation="horizontal")
            ax_arr[0].set_title("Difference")
            p = ax_arr[1].imshow(normed_gain, cmap='seismic', vmin=-1, vmax=1)
            f.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            ax_arr[1].set_title("Gain")

            import pdb
            pdb.set_trace()
            plt.close('all')

        e_loss = e_loss / len(val_data_loader)
        e_iou = e_iou / len(val_data_loader)
        print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
