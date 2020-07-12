# ---------------------------------------------------------------------------------------
# Get or view predictions of a trained model on the pathfinder dataset
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset_pathfinder
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import utils


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

    with torch.no_grad():
        for iteration, data_loader_out in enumerate(data_loader, 1):

            img, label, individual_contours_labels, full_labels, _, _ = data_loader_out

            img = img.to(device)
            label = label.to(device)

            label_out = net(img)

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

            print("image {}:  pred {:0.2f}, GT {}, net acc = {:0.2f}, loss={:0.2f}".format(
                iteration, torch.sigmoid(label_out).item(),
                label.item(), e_acc/iteration, total_loss
            ))

            # # DEBUG : view the predictions
            # # -------------------------------------------------------------------------------
            # fig, ax_arr = plt.subplots(1, 3, figsize=(15, 5))

            # # Image
            # display_img = np.squeeze(img, axis=0)
            # display_img = np.transpose(display_img, axes=(1, 2, 0))
            # display_img = \
            #     (display_img - display_img.min() / (display_img.max() - display_img.min()))
            #
            # ax_arr[0].imshow(display_img)
            # ax_arr[1].imshow(np.squeeze(individual_contours_labels))
            # ax_arr[2].imshow(np.squeeze(full_labels))
            #
            # fig.suptitle("Model Pred {:0.2f}. GT ={}".format(
            #     torch.sigmoid(label_out).item(),
            #     label.item()))
            #
            # import pdb
            # pdb.set_trace()

    e_loss = e_loss / len(data_loader)
    e_acc = e_acc / len(data_loader)

    # -----------------------------------------------------------------------------------
    #  End
    # -----------------------------------------------------------------------------------
    print("Processing took {}".format(datetime.now() - data_load_start_time))
    import pdb
    pdb.set_trace()
