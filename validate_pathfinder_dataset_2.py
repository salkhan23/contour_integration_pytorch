# ---------------------------------------------------------------------------------------
# Get or view predictions of a trained model on the pathfinder dataset
#
# NOTE: This is similar to validate pathfinder_dataset.py but also gets predictions from the
# individual contours and full label images
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader

import dataset_pathfinder
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import utils


def convert_label_to_input_image(in_label):
    """ Labels are single channel while the input is 3 dimensional
        NOTE: this is an in place operation
    """

    # Convert to binary [0, 1] mask
    # Weird scaling issue when the labels were generated (are loaded) ?
    # For individual contours the second contour has a different value so it has three unique
    # values. Even for full labels this scaling is needed.
    in_label[in_label >= 0.25] = 1
    in_label[in_label < 0.25] = 0

    # increase dimensionality to match input
    # assumes first channel (0) is for batch size
    in_label = torch.cat((in_label, in_label, in_label), 1)

    return in_label


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 5
    data_set_dir = './data/pathfinder_natural_images/test'

    save_predictions = True

    # Model
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    saved_model = \
        './results/pathfinder/' \
        'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20200629_195058_lr_1e-4_reg_1e-5/' \
        'best_accuracy.pth'

    # Immutable
    # ---------
    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))

    results_store_dir = os.path.dirname(saved_model)
    results_store_file = os.path.join(results_store_dir, 'test_data_predictions.txt')
    if os.path.exists(results_store_file):
        ans = input("Results store File {} already exists.overwrite ?".format(results_store_file))
        if 'y' not in ans.lower():
            sys.exit()
    file_handle = open(results_store_file, 'w+')
    print("Predicts will be stored @ {}".format(results_store_file))

    # -----------------------------------------------------------------------------------
    #  Data loader
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_load_start_time = datetime.now()

    print("Data Source: {}".format(data_set_dir))

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ]
    pre_process_transforms = transforms.Compose(transforms_list)

    data_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=data_set_dir,
        transform=pre_process_transforms,
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    print("Data loading Took {}. # Images {}".format(
        datetime.now() - data_load_start_time,
        len(data_loader) * 1))

    # -------------------------------------------------------------------------------
    # Loss / optimizer
    # -------------------------------------------------------------------------------
    use_gaussian_reg_on_lateral_kernels = True
    gaussian_kernel_sigma = 10
    gaussian_reg_weight = 0.000001
    lambda1 = gaussian_reg_weight

    criterion = nn.BCEWithLogitsLoss().to(device)

    if use_gaussian_reg_on_lateral_kernels:
        gaussian_mask_e = 1 - utils.get_2d_gaussian_kernel(
            net.contour_integration_layer.lateral_e.weight.shape[2:],
            sigma=gaussian_kernel_sigma)
        gaussian_mask_i = 1 - utils.get_2d_gaussian_kernel(
            net.contour_integration_layer.lateral_i.weight.shape[2:],
            sigma=gaussian_kernel_sigma)

        gaussian_mask_e = torch.from_numpy(gaussian_mask_e).float().to(device)
        gaussian_mask_i = torch.from_numpy(gaussian_mask_i).float().to(device)


        def inverse_gaussian_regularization(weight_e, weight_i):
            loss1 = (gaussian_mask_e * weight_e).abs().sum() + \
                    (gaussian_mask_i * weight_i).abs().sum()
            # print("Loss1: {:0.4f}".format(loss1))

            return loss1

    def binary_acc(y_pred, y_target):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_target.float()).sum().float()
        acc = correct_results_sum / y_target.shape[0]
        acc = torch.round(acc * 100)

        return acc
    # -------------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------------
    net.eval()
    e_loss = 0
    e_acc = 0
    e_acc_indv_contours = 0  # Accuracy when input is just the two contours
    e_acc_full_labels = 0  # Accuracy when input is the full label with end points

    file_handle.write(
        "[img_idx, ground_truth,\n"
        "input_img_raw_pred, input_img_pred, input_img_acc,\n"
        "indv_contour_raw_pred, indv_contour_pred, indv_contour_acc,\n"
        "full_labels_raw_pred, full_labels_pred, full_labels_acc]\n")
    file_handle.write("{}\n".format('-'*80))

    with torch.no_grad():
        for iteration, data_loader_out in enumerate(data_loader, 1):

            img, label, individual_contours_labels, full_labels, _, _ = data_loader_out

            img = img.to(device)
            label = label.to(device)

            label_out = net(img)
            pred = torch.round(torch.sigmoid(label_out))

            bce_loss = criterion(label_out, label.float())
            reg_loss = 0

            if use_gaussian_reg_on_lateral_kernels:
                reg_loss = \
                    inverse_gaussian_regularization(
                        net.contour_integration_layer.lateral_e.weight,
                        net.contour_integration_layer.lateral_i.weight
                    )

            total_loss = bce_loss + lambda1 * reg_loss
            accuracy = binary_acc(label_out, label)

            e_loss += total_loss.item()
            e_acc += accuracy.item()

            # print("image {}:  pred {:0.2f}, GT {}, net acc = {:0.2f}, loss={:0.2f}".format(
            #     iteration, torch.sigmoid(label_out).item(),
            #     label.item(), e_acc/iteration, total_loss
            # ))

            # Process individual contours image
            # ---------------------------------
            individual_contours_labels = convert_label_to_input_image(individual_contours_labels)
            # # more natural image like input
            # individual_contours_labels = individual_contours_labels * img

            # Normalize
            img_mean = individual_contours_labels.mean(axis=(0, 2, 3))
            img_std = individual_contours_labels.std(axis=(0, 2, 3))
            transform_functional.normalize(
                torch.squeeze(individual_contours_labels), ch_mean, ch_std, inplace=True)

            individual_contours_labels = individual_contours_labels.to(device)
            label_out_indv_contours = net(individual_contours_labels)
            pred_indv_contours = torch.round(torch.sigmoid(label_out_indv_contours))

            acc1 = binary_acc(label_out_indv_contours, label)
            e_acc_indv_contours += acc1.item()

            # Process Full labels
            # --------------------------------------------
            full_labels = convert_label_to_input_image(full_labels)
            # # more natural image like input
            # full_labels = full_labels * img

            # Normalize Input image
            transform_functional.normalize(
                torch.squeeze(full_labels), ch_mean, ch_std, inplace=True)

            full_labels = full_labels.to(device)
            label_out_full_labels = net(full_labels)
            pred_full_label = torch.round(torch.sigmoid(label_out_full_labels))

            acc2 = binary_acc(label_out_full_labels, label)
            e_acc_full_labels += acc2.item()

            print("[{}]: GT= {}, Raw [{:.2f}, {:.2f}, {:.2f}], Predictions [{}, {}, {}]. "
                  "Net [{:0.2f}, {:0.2f}, {:0.2f}]".format(
                    iteration, label.item(),
                    torch.sigmoid(label_out).item(),
                    torch.sigmoid(label_out_indv_contours).item(),
                    torch.sigmoid(label_out_full_labels).item(),
                    pred.item(),
                    pred_indv_contours.item(),
                    pred_full_label.item(),
                    e_acc / iteration,
                    e_acc_indv_contours / iteration,
                    e_acc_full_labels / iteration))

            # Store the results
            # Store the results
            file_handle.write(
                "[{:>5}, {}, {:.4f}, {}, {:6.2f}, {:.4f}, {}, {:6.2f}, "
                "{:.4f}, {}, {:6.2f}],\n".format(
                    iteration,
                    label.item(),
                    torch.sigmoid(label_out).item(),
                    pred.item(),
                    e_acc / iteration,
                    torch.sigmoid(label_out_indv_contours).item(),
                    pred_indv_contours.item(),
                    e_acc_indv_contours / iteration,
                    torch.sigmoid(label_out_full_labels).item(),
                    pred_full_label.item(),
                    e_acc_full_labels / iteration))

            # # Debug: view the predictions
            # # -----------------------------------------------
            # fig, ax_arr = plt.subplots(1, 3, figsize=(15, 5))
            #
            # # Image
            # display_img = np.squeeze(img, axis=0)
            # display_img = np.transpose(display_img, axes=(1, 2, 0))
            # display_img = \
            #     (display_img - display_img.min() / (display_img.max() - display_img.min()))
            #
            # ax_arr[0].imshow(display_img)
            # ax_arr[0].set_title("Pred: {:0.2f}".format(torch.sigmoid(label_out).item()))
            #
            # ax_arr[1].imshow(np.transpose(np.squeeze(individual_contours_labels), axes=(1, 2, 0)))
            # ax_arr[1].set_title("Pred: {:0.2f}".format(
            #     torch.sigmoid(label_out_indv_contours).item()))
            #
            # ax_arr[2].imshow(np.transpose(np.squeeze(full_labels), axes=(1, 2, 0)))
            # ax_arr[2].set_title("Pred: {:0.2f}".format(
            #     torch.sigmoid(label_out_full_labels).item()))
            #
            # fig.suptitle(
            #     "Image [{}]. GT = {}. Net accuracies[ Input img {:0.2f}, "
            #     "Individual Labels {:0.2f}, Full labels {:0.2f}]".format(
            #         iteration,
            #         label.item(),
            #         e_acc / iteration,
            #         e_acc_indv_contours / iteration,
            #         e_acc_full_labels / iteration))
            #
            # import pdb
            # pdb.set_trace()

    e_loss = e_loss / len(data_loader)
    e_acc = e_acc / len(data_loader)
    e_acc_full_labels = e_acc_full_labels / len(data_loader)
    e_acc_indv_contours = e_acc_indv_contours / len(data_loader)

    # -----------------------------------------------------------------------------------
    #  End
    # -----------------------------------------------------------------------------------
    print("Processing took {}".format(datetime.now() - data_load_start_time))
    file_handle.close()
    import pdb
    pdb.set_trace()
