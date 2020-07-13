# ---------------------------------------------------------------------------------------
#  Train a model with two heads on the pathfinder and contour dataset jointly
# ---------------------------------------------------------------------------------------
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import utils
import models.new_piech_models as new_piech_models
import dataset as dataset_contour
import dataset_pathfinder
import torch.nn.init

import experiment_gain_vs_len
import experiment_gain_vs_spacing


class JointPathfinderContourResnet50(nn.Module):
    """
    """
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True):
        super(JointPathfinderContourResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(
            pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            torch.nn.init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        self.contour_integration_layer = contour_integration_layer

        self.pathfinder_classifier = \
            new_piech_models.BinaryClassifier(n_in_channels=self.num_edge_extract_chan)

        self.contour_classifier = new_piech_models.ClassifierHead(
            num_channels=self.num_edge_extract_chan)

    def forward(self, in_img):

        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.contour_integration_layer(x)

        out_contour = self.contour_classifier(x)
        out_pathfinder = self.pathfinder_classifier(x)

        return out_contour, out_pathfinder


def binary_acc(y_pred, y_target):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_target.float()).sum().float()
    acc = correct_results_sum / y_target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7
    base_results_store_dir = './results'

    # Immutable
    # ---------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    plt.ion()
    start_time = datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    model = JointPathfinderContourResnet50(cont_int_layer)

    model = model.to(device)

    print("Name: {}".format(model.__class__.__name__))
    print(model)

    # Get name of contour integration layer
    temp = vars(model)  # Returns a dictionary.
    layers = temp['_modules']  # Returns all top level modules (layers)
    cont_int_layer_type = ''
    if 'contour_integration_layer' in layers:
        cont_int_layer_type = model.contour_integration_layer.__class__.__name__

    results_store_dir = os.path.join(
        base_results_store_dir,
        model.__class__.__name__ + '_' + cont_int_layer_type +
        datetime.now().strftime("_%Y%m%d_%H%M%S"))

    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -------------------------------------------------------------------------------
    # Data Loaders
    # -------------------------------------------------------------------------------
    print("====> Setting up Data loaders ")
    pathfinder_data_set_dir = './data/pathfinder_natural_images'

    contour_data_set_dir = './data/channel_wise_optimal_full14_frag7'
    c_len_arr = None
    beta_arr = None
    alpha_arr = None
    gabor_set_arr = None
    contour_data_set_train_subset_size = None
    contour_data_set_test_subset_size = None

    train_batch_size = 32
    test_batch_size = 1

    # Pathfinder
    # -----------------------------------------------------------------------------------
    print("Setting up Pathfinder Data loaders... ")
    data_load_start_time = datetime.now()
    print("Data Source: {}".format(pathfinder_data_set_dir))

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    pathfinder_transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ]
    pathfinder_pre_process_transforms = transforms.Compose(pathfinder_transforms_list)

    pathfinder_train_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=os.path.join(pathfinder_data_set_dir, 'train'),
        transform=pathfinder_pre_process_transforms,
    )

    pathfinder_train_data_loader = DataLoader(
        dataset=pathfinder_train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    pathfinder_val_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=os.path.join(pathfinder_data_set_dir, 'test'),
        transform=pathfinder_pre_process_transforms,
    )

    pathfinder_val_data_loader = DataLoader(
        dataset=pathfinder_val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True
    )
    print("Pathfinder Data loading Took {}. # Train {}, # Test {}".format(
        datetime.now() - data_load_start_time,
        len(pathfinder_train_data_loader) * train_batch_size,
        len(pathfinder_val_data_loader) * test_batch_size
    ))

    # Contour Dataset
    # -----------------------------------------------------------------------------------
    print("Setting up Contour Data loaders... ")
    data_load_start_time = datetime.now()
    print("Data Source: {}".format(contour_data_set_dir))
    print("Restrictions:\n clen={},\n beta={},\n alpha={},\n gabor_sets={},\n "
          "train_subset={},\ntest subset={}\n".format(
            c_len_arr, beta_arr, alpha_arr, gabor_set_arr, contour_data_set_train_subset_size, contour_data_set_test_subset_size))

    # get mean/std of dataset
    meta_data_file = os.path.join(contour_data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as file_handle:
        meta_data = pickle.load(file_handle)

    # Pre-processing
    contour_transforms_list = [
        transforms.Normalize(mean=meta_data['channel_mean'], std=meta_data['channel_std']),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ]
    contour_pre_process_transforms = transforms.Compose(contour_transforms_list)

    contour_train_set = dataset_contour.Fields1993(
        data_dir=os.path.join(contour_data_set_dir, 'train'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=contour_pre_process_transforms,
        subset_size=contour_data_set_train_subset_size,
        c_len_arr=c_len_arr,
        beta_arr=beta_arr,
        alpha_arr=alpha_arr,
        gabor_set_arr=gabor_set_arr
    )

    contour_train_data_loader = DataLoader(
        dataset=contour_train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    contour_val_set = dataset_contour.Fields1993(
        data_dir=os.path.join(contour_data_set_dir, 'val'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=contour_pre_process_transforms,
        subset_size=contour_data_set_test_subset_size,
        c_len_arr=c_len_arr,
        beta_arr=beta_arr,
        alpha_arr=alpha_arr,
        gabor_set_arr=gabor_set_arr
    )

    contour_val_data_loader = DataLoader(
        dataset=contour_val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True
    )

    print("Contour Data loading Took {}. # Train {}, # Test {}".format(
        datetime.now() - data_load_start_time,
        len(contour_train_data_loader) * train_batch_size,
        len(contour_val_data_loader) * test_batch_size
    ))

    # -----------------------------------------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------------------------------------
    learning_rate = 1e-4
    use_gaussian_reg_on_lateral_kernels = True
    gaussian_kernel_sigma = 10
    gaussian_reg_weight = 1e-5
    gaussian_reg_sigma = 10
    lambda1 = gaussian_reg_weight
    detect_thres = 0.5

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss().to(device)

    if use_gaussian_reg_on_lateral_kernels:
        gaussian_mask_e = 1 - utils.get_2d_gaussian_kernel(
            model.contour_integration_layer.lateral_e.weight.shape[2:],
            sigma=gaussian_kernel_sigma)
        gaussian_mask_i = 1 - utils.get_2d_gaussian_kernel(
            model.contour_integration_layer.lateral_i.weight.shape[2:],
            sigma=gaussian_kernel_sigma)

        gaussian_mask_e = torch.from_numpy(gaussian_mask_e).float().to(device)
        gaussian_mask_i = torch.from_numpy(gaussian_mask_i).float().to(device)


        def inverse_gaussian_regularization(weight_e, weight_i):
            loss1 = (gaussian_mask_e * weight_e).abs().sum() + \
                    (gaussian_mask_i * weight_i).abs().sum()
            # print("Loss1: {:0.4f}".format(loss1))

            return loss1

    # -----------------------------------------------------------------------------------
    #  Training Validation Routines
    # -----------------------------------------------------------------------------------
    def train_pathfinder():
        """ Train for one Epoch over the train data set """
        model.train()
        e_loss = 0
        e_acc = 0

        for iteration, data_loader_out in enumerate(pathfinder_train_data_loader, 1):
            optimizer.zero_grad()  # zero the parameter gradients

            img, label, _, _, _, _ = data_loader_out

            img = img.to(device)
            label = label.to(device)

            # Second part is pathfinder out
            _, label_out = model(img)

            bce_loss = criterion(label_out, label.float())
            reg_loss = 0

            if use_gaussian_reg_on_lateral_kernels:
                reg_loss = \
                    inverse_gaussian_regularization(
                        model.contour_integration_layer.lateral_e.weight,
                        model.contour_integration_layer.lateral_i.weight
                    )

            total_loss = bce_loss + lambda1 * reg_loss
            acc = binary_acc(label_out, label)

            # print("Loss: {:0.4f}, bce_loss {:0.4f}, lateral kernels reg_loss {:0.4f}, "
            #       "acc {:0.4f}".format(total_loss, bce_loss,  lambda1 * reg_loss, acc))

            total_loss.backward()
            optimizer.step()

            e_loss += total_loss.item()
            e_acc += acc.item()

        e_loss = e_loss / len(pathfinder_train_data_loader)
        e_acc = e_acc / len(pathfinder_train_data_loader)

        # iou_arr = ["{:0.2f}".format(item) for item in e_iou]
        # print("Train Epoch {} Loss = {:0.4f}, Acc = {}".format(epoch, e_loss, e_acc))

        return e_loss, e_acc

    def validate_pathfinder():
        """ Get loss over validation set """
        model.eval()
        e_loss = 0
        e_acc = 0

        with torch.no_grad():
            for iteration, data_loader_out in enumerate(pathfinder_val_data_loader, 1):

                img, label, _, _, _, _ = data_loader_out

                img = img.to(device)
                label = label.to(device)

                _, label_out = model(img)

                bce_loss = criterion(label_out, label.float())
                reg_loss = 0

                if use_gaussian_reg_on_lateral_kernels:
                    reg_loss = \
                        inverse_gaussian_regularization(
                            model.contour_integration_layer.lateral_e.weight,
                            model.contour_integration_layer.lateral_i.weight
                        )

                total_loss = bce_loss + lambda1 * reg_loss
                acc = binary_acc(label_out, label)

                e_loss += total_loss.item()
                e_acc += acc.item()

        e_loss = e_loss / len(pathfinder_val_data_loader)
        e_acc = e_acc / len(pathfinder_val_data_loader)

        # print("Val Loss = {:0.4f}, Accuracy={}".format(e_loss, e_acc))

        return e_loss, e_acc

    def train_contour():
        """ Train for one Epoch over the train data set """
        model.train()
        e_loss = 0
        e_iou = 0

        for iteration, (img, label) in enumerate(contour_train_data_loader, 1):
            optimizer.zero_grad()  # zero the parameter gradients

            img = img.to(device)
            label = label.to(device)

            label_out, _ = model(img)
            batch_loss = criterion(label_out, label.float())

            kernel_loss = \
                inverse_gaussian_regularization(
                    model.contour_integration_layer.lateral_e.weight,
                    model.contour_integration_layer.lateral_i.weight
                )

            total_loss = batch_loss + lambda1 * kernel_loss

            # print("Total Loss: {:0.4f}, cross_entropy_loss {:0.4f}, kernel_loss {:0.4f}".format(
            #     total_loss, batch_loss,  lambda1 * kernel_loss))
            #
            # import pdb
            # pdb.set_trace()

            total_loss.backward()
            optimizer.step()

            e_loss += total_loss.item()

            preds = (torch.sigmoid(label_out) > detect_thres)
            e_iou += utils.intersection_over_union(
                preds.float(), label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(contour_train_data_loader)
        e_iou = e_iou / len(contour_train_data_loader)

        # print("Train Epoch {} Loss = {:0.4f}, IoU={:0.4f}".format(epoch, e_loss, e_iou))

        return e_loss, e_iou

    def validate_contour():
        """ Get loss over validation set """
        model.eval()
        e_loss = 0
        e_iou = 0

        with torch.no_grad():
            for iteration, (img, label) in enumerate(contour_val_data_loader, 1):
                img = img.to(device)
                label = label.to(device)

                label_out, _ = model(img)
                batch_loss = criterion(label_out, label.float())

                kernel_loss = \
                    inverse_gaussian_regularization(
                        model.contour_integration_layer.lateral_e.weight,
                        model.contour_integration_layer.lateral_i.weight
                    )

                total_loss = batch_loss + lambda1 * kernel_loss

                e_loss += total_loss.item()
                preds = (torch.sigmoid(label_out) > detect_thres)
                e_iou += utils.intersection_over_union(
                    preds.float(), label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(contour_val_data_loader)
        e_iou = e_iou / len(contour_val_data_loader)

        # print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

        return e_loss, e_iou

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("====> Starting Training ")
    training_start_time = datetime.now()

    num_epochs = 10
    best_acc = 0

    pathfinder_train_history = []
    pathfinder_val_history = []
    contour_train_history = []
    contour_val_history = []

    lr_history = []

    for epoch in range(0, num_epochs):

        epoch_start_time = datetime.now()

        pathfinder_train_history.append(train_pathfinder())
        pathfinder_val_history.append(validate_pathfinder())

        print("Epoch [{}/{}], Train: loss={:0.4f}, Acc={:0.2f}. Val: loss={:0.4f}, Acc={:0.2f}."
              " Time {}".format(
                epoch + 1, num_epochs,
                pathfinder_train_history[epoch][0],
                pathfinder_train_history[epoch][1],
                pathfinder_val_history[epoch][0],
                pathfinder_val_history[epoch][1],
                datetime.now() - epoch_start_time))

        epoch_start_time_contour = datetime.now()
        contour_train_history.append(train_contour())
        contour_val_history.append(validate_contour)

        print("Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, IoU={:0.4f}. "
              "Time {}".format(
                epoch + 1, num_epochs,
                contour_train_history[epoch][0],
                contour_train_history[epoch][1],
                contour_val_history[epoch][0],
                contour_val_history[epoch][1],
                datetime.now() - epoch_start_time))

        lr_history.append(get_lr(optimizer))
        lr_scheduler.step(epoch)

    training_time = datetime.now() - training_start_time
    print('Finished Training. Training took {}'.format(training_time))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
