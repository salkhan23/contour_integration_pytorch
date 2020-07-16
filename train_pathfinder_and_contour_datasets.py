# ---------------------------------------------------------------------------------------
# Train a model on the pathfinder and contour Dataset jointly
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


def binary_acc(y_pred, y_target):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_target.float()).sum().float()
    acc = correct_results_sum / y_target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def plot_pathfinder_results(train_history, val_history, result_dir):
    train_history = np.array(train_history)
    val_history = np.array(val_history)

    f, ax_arr = plt.subplots(1, 2)

    n_epochs = train_history[-1, 0]

    ax_arr[0].plot(np.arange(1, n_epochs + 1), train_history[:, 0], label='train')
    ax_arr[0].plot(np.arange(1, n_epochs + 1), val_history[:, 0], label='val')
    ax_arr[0].set_xlabel('Epoch')
    ax_arr[0].set_title("Loss Vs Time")
    ax_arr[0].grid(True)
    ax_arr[0].legend()

    ax_arr[1].plot(np.arange(1, n_epochs + 1), train_history[:, 1], label='train')
    ax_arr[1].plot(np.arange(1, n_epochs + 1), val_history[:, 1], label='val')
    ax_arr[1].set_xlabel('Epoch')
    ax_arr[1].set_title("Accuracy Vs Time")
    ax_arr[1].grid(True)
    ax_arr[1].legend()

    f.suptitle('Pathfinder Task')
    f.savefig(os.path.join(result_dir, 'pathfinder_loss_and accuracy.jpg'), format='jpg')


def plot_contour_results(train_history, val_history, results_dir):

    train_history = np.array(train_history)
    val_history = np.array(val_history)

    f, ax_arr = plt.subplots(1, 2)

    ax_arr[0].plot(train_history[:, 0], label='train')
    ax_arr[0].plot(val_history[:, 0], label='validation')
    ax_arr[0].set_xlabel("Epoch")
    ax_arr[0].set_title("Loss vs Time")
    ax_arr[0].grid(True)
    ax_arr[0].legend()

    ax_arr[0].plot(train_history[:, 1], label='train')
    ax_arr[0].plot(val_history[:, 1], label='validation')
    ax_arr[0].set_xlabel('Epoch')
    ax_arr[0].set_title("IoU vs Time")
    ax_arr[0].grid(True)
    ax_arr[0].legend()

    f.suptitle("Contour Detection Task")
    f.savefig(os.path.join(results_dir, 'contour_loss_and_iou.jpg'), format='jpg')


def main(model, train_params, data_set_params, base_results_store_dir='./results'):
    # -----------------------------------------------------------------------------------
    # Validate Parameters
    # -----------------------------------------------------------------------------------
    print("====> Validating Parameters ")
    # Pathfinder Dataset
    # ------------------
    pathfinder_required_data_set_params = ['pathfinder_data_set_dir']
    for key in pathfinder_required_data_set_params:
        assert key in data_set_params, 'data_set_params does not have required key {}'.format(key)
    pathfinder_data_set_dir = data_set_params['pathfinder_data_set_dir']

    # Optional
    pathfinder_train_subset_size = data_set_params.get('pathfinder_train_subset_size', None)
    pathfinder_test_subset_size = data_set_params.get('pathfinder_test_subset_size', None)

    # Contour Dataset
    # ---------------
    contour_required_data_set_params = ['contour_data_set_dir']
    for key in contour_required_data_set_params:
        assert key in data_set_params, 'data_set_params does not have required key {}'.format(key)
    contour_data_set_dir = data_set_params['contour_data_set_dir']

    # Optional
    contour_train_subset_size = data_set_params.get('contour_train_subset_size', None)
    contour_test_subset_size = data_set_params.get('contour_test_subset_size', None)
    c_len_arr = data_set_params.get('c_len_arr', None)
    beta_arr = data_set_params.get('beta_arr', None)
    alpha_arr = data_set_params.get('alpha_arr', None)
    gabor_set_arr = data_set_params.get('gabor_set_arr', None)

    # Training
    # --------
    required_training_params = \
        ['train_batch_size', 'test_batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_params:
        assert key in train_params, 'training_params does not have required key {}'.format(key)
    train_batch_size = train_params['train_batch_size']
    test_batch_size = train_params['test_batch_size']
    learning_rate = train_params['learning_rate']
    num_epochs = train_params['num_epochs']

    # Optional
    lambda1 = train_params.get('gaussian_reg_weight', 0)
    gaussian_kernel_sigma = train_params.get('gaussian_reg_sigma', 0)
    use_gaussian_reg_on_lateral_kernels = False
    if lambda1 is not 0 and gaussian_kernel_sigma is not 0:
        use_gaussian_reg_on_lateral_kernels = True

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
    print("Name: {}".format(model.__class__.__name__))
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get name of contour integration layer
    temp = vars(model)  # Returns a dictionary.
    layers = temp['_modules']  # Returns all top level modules (layers)
    cont_int_layer_type = ''
    if 'contour_integration_layer' in layers:
        cont_int_layer_type = model.contour_integration_layer.__class__.__name__

    # Actual Results store directory
    results_store_dir = os.path.join(
        base_results_store_dir,
        model.__class__.__name__ + '_' + cont_int_layer_type +
        datetime.now().strftime("_%Y%m%d_%H%M%S"))

    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # -----------------------------------------------------------------------------------
    # Data Loaders
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_load_start_time = datetime.now()

    # Pathfinder
    # --------------------------------------
    print("Setting up Pathfinder Data loaders... ")
    print("Data Source: {}".format(pathfinder_data_set_dir))

    # Pre-processing
    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    pathfinder_transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std),
        # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
    ]
    pathfinder_pre_process_transforms = transforms.Compose(pathfinder_transforms_list)

    pathfinder_train_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=os.path.join(pathfinder_data_set_dir, 'train'),
        transform=pathfinder_pre_process_transforms,
        subset_size=pathfinder_train_subset_size,
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
        subset_size=pathfinder_test_subset_size
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
    # ---------------
    print("Setting up Contour Data loaders... ")
    data_load_start_time = datetime.now()
    print("Data Source: {}".format(contour_data_set_dir))
    print("\tRestrictions:\n\t clen={},\n\t beta={},\n\t alpha={},\n\t gabor_sets={},\n\t "
          "train_subset={},\n\ttest subset={}\n".format(
            c_len_arr, beta_arr, alpha_arr, gabor_set_arr, contour_train_subset_size,
            contour_test_subset_size))

    # Get mean/std of dataset
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
        subset_size=contour_train_subset_size,
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
        subset_size=contour_test_subset_size,
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
    detect_thres = 0.5

    optimizer = optim.Adam(
        filter(lambda p1: p1.requires_grad, model.parameters()),
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

    def write_training_and_model_details(f_handle):
        # Dataset Parameters:
        f_handle.write("Data Set Parameters {}\n".format('-' * 60))
        f_handle.write("CONTOUR DATASET \n")
        f_handle.write("Source                   : {}\n".format(contour_data_set_dir))
        f_handle.write("Restrictions             :\n")
        f_handle.write("  Lengths                : {}\n".format(c_len_arr))
        f_handle.write("  Beta                   : {}\n".format(beta_arr))
        f_handle.write("  Alpha                  : {}\n".format(alpha_arr))
        f_handle.write("  Gabor Sets             : {}\n".format(gabor_set_arr))
        f_handle.write("  Train subset size      : {}\n".format(contour_train_subset_size))
        f_handle.write("  Test subset size       : {}\n".format(contour_test_subset_size))
        f_handle.write("Number of Images         : Train {}, Test {}".format(
            len(contour_train_set.images), len(contour_val_set.images)))
        f_handle.write("Train Set Mean {}, std {}\n".format(
            contour_train_set.data_set_mean, contour_train_set.data_set_std))
        f_handle.write("Val   Set Mean {}, std {}\n".format(
            contour_val_set.data_set_mean, contour_val_set.data_set_std))

        f_handle.write("PATHFINDER  DATASET\n")
        f_handle.write("Source                   : {}\n".format(pathfinder_data_set_dir))
        f_handle.write("Restrictions             :\n")
        f_handle.write("  Train subset size      : {}\n".format(pathfinder_train_subset_size))
        f_handle.write("  Test subset size       : {}\n".format(pathfinder_test_subset_size))
        f_handle.write("Number of Images         : Train {}, Test {}".format(
            len(pathfinder_train_set.images), len(pathfinder_val_set.images)))

        # Training Parameters:
        f_handle.write("Training Parameters {}\n".format('-' * 60))
        f_handle.write("Train batch size         : {}\n".format(train_batch_size))
        f_handle.write("Val batch size           : {}\n".format(test_batch_size))
        f_handle.write("Epochs                   : {}\n".format(num_epochs))
        f_handle.write("Optimizer                : {}\n".format(optimizer.__class__.__name__))
        f_handle.write("learning rate            : {}\n".format(learning_rate))
        f_handle.write("Loss Fcn                 : {}\n".format(criterion.__class__.__name__))
        f_handle.write("Gaussian Regularization on lateral kernels: {}\n".format(
            use_gaussian_reg_on_lateral_kernels))
        if use_gaussian_reg_on_lateral_kernels:
            f_handle.write("  Gaussian Reg. sigma    : {}\n".format(
                gaussian_kernel_sigma))
            f_handle.write("  Gaussian Reg. weight   : {}\n".format(lambda1))
        f_handle.write("IoU Detection Threshold  : {}\n".format(detect_thres))

        f_handle.write("Image pre-processing :\n")
        f_handle.write("Contour Dataset:\n")
        print(contour_pre_process_transforms, file=f_handle)
        f_handle.write("Pathfinder Dataset:\n")
        print(pathfinder_pre_process_transforms, file=f_handle)

        # Model Details
        f_handle.write("Model Parameters {}\n".format('-' * 63))
        f_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
        f_handle.write("\n")
        print(model, file=file_handle)

        tmp = vars(model)  # Returns a dictionary.
        p = [item for item in tmp if not item.startswith('_')]
        for var in sorted(p):
            f_handle.write("{}: {}\n".format(var, getattr(model, var)))

        layers1 = tmp['_modules']  # Returns all top level modules (layers)
        if 'contour_integration_layer' in layers1:

            f_handle.write("Contour Integration Layer: {}\n".format(
                model.contour_integration_layer.__class__.__name__))

            # print fixed hyper parameters
            f_handle.write("Hyper parameters\n")

            cont_int_layer_vars = \
                [item for item in vars(model.contour_integration_layer) if not item.startswith('_')]
            for var in sorted(cont_int_layer_vars):
                f_handle.write("\t{}: {}\n".format(
                    var, getattr(model.contour_integration_layer, var)))

            # print parameter names and whether they are trainable
            f_handle.write("Contour Integration Layer Parameters\n")
            layer_params = vars(model.contour_integration_layer)['_parameters']
            for k, v in sorted(layer_params.items()):
                f_handle.write("\t{}: requires_grad {}\n".format(k, v.requires_grad))

        # Headers for columns in training details in summary file
        f_handle.write("{}\n".format('-' * 80))
        f_handle.write("Training details\n")
        f_handle.write("[Epoch,\ncontour train_loss, train_iou, val_loss, val_iou\n")
        f_handle.write("pathfinder train_loss, train_acc, val_loss, val_acc]\n")

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("====> Starting Training ")
    training_start_time = datetime.now()

    pathfinder_train_history = []
    pathfinder_val_history = []
    contour_train_history = []
    contour_val_history = []
    lr_history = []

    # Summary File
    # ------------
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    file_handle = open(summary_file, 'w+')

    write_training_and_model_details(file_handle)

    # Actual main loop start
    # ----------------------
    print("train_batch_size={}, test_batch_size= {}, lr={}, epochs={}".format(
        train_batch_size, test_batch_size, learning_rate, num_epochs))

    for epoch in range(0, num_epochs):

        # Contour Dataset First
        epoch_start_time = datetime.now()
        contour_train_history.append(train_contour())
        contour_val_history.append(validate_contour())

        print("Epoch [{}/{}], Contour    Train: loss={:0.4f}, IoU={:0.4f}. Val: "
              "loss={:0.4f}, IoU={:0.4f}. Time {}".format(
                epoch + 1, num_epochs,
                contour_train_history[epoch][0],
                contour_train_history[epoch][1],
                contour_val_history[epoch][0],
                contour_val_history[epoch][1],
                datetime.now() - epoch_start_time))

        # Pathfinder Dataset
        epoch_start_time = datetime.now()
        pathfinder_train_history.append(train_pathfinder())
        pathfinder_val_history.append(validate_pathfinder())

        print("Epoch [{}/{}], Pathfinder Train: loss={:0.4f}, Acc={:0.3f}. Val: "
              "loss={:0.4f}, Acc={:0.3f}. Time {}".format(
                epoch + 1, num_epochs,
                pathfinder_train_history[epoch][0],
                pathfinder_train_history[epoch][1],
                pathfinder_val_history[epoch][0],
                pathfinder_val_history[epoch][1],
                datetime.now() - epoch_start_time))

        lr_history.append(get_lr(optimizer))
        lr_scheduler.step(epoch)

        # Save Last epoch weights
        torch.save(
            model.state_dict(),
            os.path.join(results_store_dir, 'last_epoch.pth')
        )

        file_handle.write(
            "[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, "
            "{:0.4f}, {:0.2f}, {:0.4f}, {:0.2f}],\n".format(
                epoch + 1,
                contour_train_history[epoch][0],
                contour_train_history[epoch][1],
                contour_val_history[epoch][0],
                contour_val_history[epoch][1],
                pathfinder_train_history[epoch][0],
                pathfinder_train_history[epoch][1],
                pathfinder_val_history[epoch][0],
                pathfinder_val_history[epoch][1]))

    training_time = datetime.now() - training_start_time
    print('Finished Training. Training took {}'.format(training_time))

    file_handle.write("{}\n".format('-' * 80))
    file_handle.write("Train Duration       : {}\n".format(training_time))
    file_handle.close()

    # -----------------------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------------------
    plot_pathfinder_results(pathfinder_train_history, pathfinder_val_history, results_store_dir)
    plot_contour_results(contour_train_history, contour_val_history, results_store_dir)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7

    data_set_parameters = {
        'pathfinder_data_set_dir': './data/pathfinder_natural_images',
        # 'pathfinder_train_subset_size': 1000,
        # 'pathfinder_test_subset_size': 10,
        'contour_data_set_dir': "./data/channel_wise_optimal_full14_frag7",
        # 'contour_train_subset_size': 1000,
        # 'contour_test_subset_size': 10,
    }

    train_parameters = {
        'train_batch_size': 16,
        'test_batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'gaussian_reg_weight': 1e-5,
        'gaussian_reg_sigma': 10,
    }

    # Immutable
    # ---------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    plt.ion()
    start_time = datetime.now()

    # Model
    # -----------------------------------------------------------------------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.JointPathfinderContourResnet50(cont_int_layer)

    main(net, train_params=train_parameters, data_set_params=data_set_parameters,
         base_results_store_dir='./results/joint_training')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
