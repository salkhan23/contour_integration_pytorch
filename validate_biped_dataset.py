# ---------------------------------------------------------------------------------------
# Get or view PREDICTIONs of a trained model on the contour integration dataset
# This is equivalent to the validate_biped_dataset.py script but for the contour dataset.
#
# validate_contour_data_set.py calculates IoU scores for different subsets of the data,
# ie. per length, per length for straight contours, per length for curved contours.
# Both scripts can be used to view predictions.
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset_biped
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import utils


edge_extract_act = []
cont_int_in_act = []
cont_int_out_act = []


def get_predictions(model, data_loader, store_dir, detect_thres=0.3):
    """
    :param model:
    :param data_loader: 
    :param store_dir:
    :param detect_thres:

    :return: None
    """
    results_dir = os.path.join(store_dir, 'predictions')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    list_of_files = data_loader.dataset.labels
    print("".format("Storing {} predictions".format(len(list_of_files))))

    model.eval()
    total_iou = 0

    with torch.no_grad():
        for iteration, (img, label) in enumerate(data_loader, 0):

            img = img.to(device)
            label = label.to(device)

            label_out = model(img)

            preds = (torch.sigmoid(label_out) > detect_thres)
            iou = utils.intersection_over_union(
                preds.float(), label.float()).cpu().detach().numpy()
            total_iou += iou

            # Before visualizing Sigmoid the output. This is already done in the loss function
            label_out = torch.sigmoid(label_out)

            plt.imsave(
                fname=os.path.join(results_dir, list_of_files[iteration].split('/')[-1]),
                arr=np.squeeze(label_out),
                cmap=plt.cm.gray,
            )

            # # -----------------------------------------------------------------------------
            # #  Plot input image, label and prediction
            # # ---------------------------------------------------------------------------
            # display_img = img.detach().cpu().numpy()
            # display_img = np.squeeze(display_img)
            # display_img = np.transpose(display_img, axes=(1, 2, 0))
            # display_img = (display_img - display_img.min()) / \
            #     (display_img.max() - display_img.min())
            #
            # f, ax_arr = plt.subplots(2, 2, figsize=(11, 9))
            #
            # ax_arr[0][0].imshow(display_img)
            # ax_arr[0][0].set_title("Image")
            #
            # label = label.detach().cpu().numpy()
            # label = np.squeeze(label)
            #
            # p = ax_arr[1][0].imshow(label)
            # # f.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            # ax_arr[1][0].set_title("GT")
            #
            # label_out = label_out.detach().cpu().numpy()
            # label_out = np.squeeze(label_out)
            #
            # p = ax_arr[0][1].imshow(label_out)
            # f.colorbar(p, ax=ax_arr[0][1], orientation="vertical")
            # ax_arr[0][1].set_title('Model Output')
            #
            # p1 = ax_arr[1][1].imshow(np.squeeze(preds))
            # # f.colorbar(p, ax=ax_arr[1][1], orientation="horizontal")
            # ax_arr[1][1].set_title('Threshold {} output. IoU={:0.4f}'.format(detect_thres, iou))
            #
            # # Plot Contour Integration Layer Input and output
            # # ---------------------------------------------------------------------------
            # # sum over all channels
            # edge_out = edge_extract_act.squeeze().max(axis=0)
            # cont_int_in = cont_int_in_act.squeeze().max(axis=0)
            # cont_int_out = cont_int_out_act.squeeze().max(axis=0)
            #
            # f2, ax_arr = plt.subplots(1, 3)
            # p = ax_arr[0].imshow(edge_out)
            # f2.colorbar(p, ax=ax_arr[0], orientation="horizontal")
            # ax_arr[0].set_title('Edge Extraction Out')
            #
            # p = ax_arr[1].imshow(cont_int_in)
            # f2.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            # ax_arr[1].set_title('Contour Integration In')
            #
            # p = ax_arr[2].imshow(cont_int_out)
            # f2.colorbar(p, ax=ax_arr[2], orientation="horizontal")
            # ax_arr[2].set_title('Contour Integration Out')
            #
            # # visualize the difference added by the contour integration layer
            # # ----------------------------------------------------------------
            # diff = cont_int_out - cont_int_in
            # gain = cont_int_out / (cont_int_in + 0.0001)
            #
            # f3, ax_arr = plt.subplots(1, 2)
            # f3.suptitle("Modifications added by the contour integration layer")
            # p = ax_arr[0].imshow(diff, cmap='seismic', vmin=-diff.max(),
            #                      vmax=diff.max())
            # f.colorbar(p, ax=ax_arr[0], orientation="horizontal")
            # ax_arr[0].set_title("Difference")
            # p = ax_arr[1].imshow(gain, cmap='seismic', vmin=-3, vmax=3)
            # f.colorbar(p, ax=ax_arr[1], orientation="horizontal")
            # ax_arr[1].set_title("Gain")
            #
            # import pdb
            # pdb.set_trace()
            # plt.close('all')

    total_iou = total_iou / len(data_loader)
    print("Mean IoU={:0.4f}".format(total_iou))


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 5
    data_set_dir = './data/BIPED/edges'

    # Build Model
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    # cont_int_layer = new_piech_models.CurrentDivisiveInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)

    cont_int_layer = new_control_models.ControlMatchParametersLayer(
         lateral_e_size=15, lateral_i_size=15)
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)
    
    saved_model = \
        'results/biped' \
        '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base' \
        '/last_epoch.pth'

    # Immutable
    # ---------------------------------------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.load_state_dict(torch.load(saved_model, map_location=device))

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    pre_process_transforms = transforms.Compose([
        transforms.Normalize(mean=ch_mean, std=ch_std),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ])

    val_set = dataset_biped.BipedDataSet(
        data_dir=data_set_dir,
        dataset_type='test',
        transform=pre_process_transforms,
        resize_size=(256, 256)
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=1,
        shuffle=False,  # Do not change needed to save predictions with correct file names
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    get_predictions(
        net,
        val_data_loader,
        store_dir=os.path.dirname(saved_model))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
