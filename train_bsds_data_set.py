# ------------------------------------------------------------------------------------
#  Train a contour integration model on BSDS Dataset
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset_bsds
import utils
import models.new_piech_models as new_piech_models

import experiment_gain_vs_len
import experiment_gain_vs_spacing


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def main(model, train_params, data_set_params, base_results_store_dir='./results'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -----------------------------------------------------------------------------------
    # Sanity Checks
    # -----------------------------------------------------------------------------------
    # Validate Data set parameters
    # ----------------------------
    required_data_set_params = ['data_set_dir']
    for key in required_data_set_params:
        assert key in data_set_params, 'data_set_params does not have required key {}'.format(key)
    data_set_dir = data_set_params['data_set_dir']

    # Validate training parameters
    # ----------------------------
    required_training_params = [
        'train_batch_size', 'test_batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_params:
        assert key in train_params, 'training_params does not have required key {}'.format(key)

    train_batch_size = train_params['train_batch_size']
    test_batch_size = train_params['test_batch_size']
    learning_rate = train_params['learning_rate']
    num_epochs = train_params['num_epochs']

    lambda1 = train_params['gaussian_reg_weight']
    gaussian_kernel_sigma = train_params['gaussian_reg_sigma']

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
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

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_load_start_time = datetime.now()

    print("Data Source: {}".format(data_set_dir))

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]
    # print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    pre_process_transforms = transforms.Compose([
        transforms.Normalize(mean=ch_mean, std=ch_std),
    ])

    train_set = dataset_bsds.BSDS(
        data_dir=data_set_dir, image_set='train', transform=pre_process_transforms)
    train_batch_size = min(train_batch_size, len(train_set))

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True
    )

    val_set = dataset_bsds.BSDS(
        data_dir=data_set_dir, image_set='val', transform=pre_process_transforms,)
    test_batch_size = min(test_batch_size, len(val_set))

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Data loading Took {}. # Train {}, # Test {}".format(
        datetime.now() - data_load_start_time,
        len(train_data_loader) * train_batch_size,
        len(val_data_loader) * test_batch_size
    ))

    # -----------------------------------------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda params: params.requires_grad, model.parameters()),
        lr=learning_rate
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss().to(device)

    gaussian_mask_e = 1 - utils.get_2d_gaussian_kernel(
        model.contour_integration_layer.lateral_e.weight.shape[2:], sigma=gaussian_kernel_sigma)
    gaussian_mask_i = 1 - utils.get_2d_gaussian_kernel(
        model.contour_integration_layer.lateral_i.weight.shape[2:], sigma=gaussian_kernel_sigma)

    gaussian_mask_e = torch.from_numpy(gaussian_mask_e).float().to(device)
    gaussian_mask_i = torch.from_numpy(gaussian_mask_i).float().to(device)

    def inverse_gaussian_regularization(weight_e, weight_i):
        loss1 = (gaussian_mask_e * weight_e).abs().sum() + (gaussian_mask_i * weight_i).abs().sum()
        # print("Loss1: {:0.4f}".format(loss1))

        return loss1

    # -----------------------------------------------------------------------------------
    #  Training Validation Routines
    # -----------------------------------------------------------------------------------
    def train():
        """ Train for one Epoch over the train data set """
        model.train()
        e_loss = 0
        e_iou = 0

        for iteration, (img, label) in enumerate(train_data_loader, 1):
            optimizer.zero_grad()  # zero the parameter gradients

            img = img.to(device)
            label = label.to(device)

            label_out = model(img)

            batch_loss = criterion(label_out, label.float())

            kernel_loss = \
                inverse_gaussian_regularization(
                    model.contour_integration_layer.lateral_e.weight,
                    model.contour_integration_layer.lateral_i.weight
                )

            total_loss = batch_loss + lambda1 * kernel_loss

            # print("Total Loss: {:0.4f}, cross_entropy_loss {:0.4f}, kernel_loss {:0.4f}".format(
            #     total_loss, batch_loss,  lambda1 * kernel_loss))

            total_loss.backward()
            optimizer.step()

            e_loss += total_loss.item()

            preds = (torch.sigmoid(label_out) > detect_thres)
            e_iou += utils.intersection_over_union(preds.float(),
                                                   label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(train_data_loader)
        e_iou = e_iou / len(train_data_loader)

        # print("Train Epoch {} Loss = {:0.4f}, IoU={:0.4f}".format(epoch, e_loss, e_iou))

        return e_loss, e_iou

    def validate():
        """ Get loss over validation set """
        model.eval()
        e_loss = 0
        e_iou = 0

        with torch.no_grad():
            for iteration, (img, label) in enumerate(val_data_loader, 1):
                img = img.to(device)
                label = label.to(device)

                label_out = model(img)
                batch_loss = criterion(label_out, label.float())

                kernel_loss = \
                    inverse_gaussian_regularization(
                        model.contour_integration_layer.lateral_e.weight,
                        model.contour_integration_layer.lateral_i.weight
                    )

                total_loss = batch_loss + lambda1 * kernel_loss

                e_loss += total_loss.item()
                preds = (torch.sigmoid(label_out) > detect_thres)
                e_iou += utils.intersection_over_union(preds.float(),
                                                       label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(val_data_loader)
        e_iou = e_iou / len(val_data_loader)

        # print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

        return e_loss, e_iou

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("====> Starting Training ")
    training_start_time = datetime.now()

    detect_thres = 0.5

    train_history = []
    val_history = []
    lr_history = []

    best_iou = 0

    # Summary file
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    file_handle = open(summary_file, 'w+')

    file_handle.write("Data Set Parameters {}\n".format('-' * 60))
    file_handle.write("Source           : {}\n".format(data_set_dir))
    file_handle.write(
        "Train Set Mean {}, std {}\n".format(train_set.data_set_mean, train_set.data_set_std))
    file_handle.write("Validation Set Mean {}, std {}\n".format(val_set.data_set_mean,
                                                                train_set.data_set_std))

    file_handle.write("Training Parameters {}\n".format('-' * 60))
    file_handle.write("Train images     : {}\n".format(len(train_set.images)))
    file_handle.write("Val images       : {}\n".format(len(val_set.images)))
    file_handle.write("Train batch size : {}\n".format(train_batch_size))
    file_handle.write("Val batch size   : {}\n".format(test_batch_size))
    file_handle.write("Epochs           : {}\n".format(num_epochs))
    file_handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
    file_handle.write("learning rate    : {}\n".format(learning_rate))
    file_handle.write("Loss Fcn         : {}\n".format(criterion.__class__.__name__))
    file_handle.write(
        "Gaussian Regularization sigma        : {}\n".format(gaussian_kernel_sigma))
    file_handle.write("Gaussian Regularization weight        : {}\n".format(lambda1))
    file_handle.write("IoU Threshold    : {}\n".format(detect_thres))
    file_handle.write("Image pre-processing :\n")
    print(pre_process_transforms, file=file_handle)

    file_handle.write("Model Parameters {}\n".format('-' * 63))
    file_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
    file_handle.write("\n")
    print(model, file=file_handle)

    temp = vars(model)  # Returns a dictionary.
    file_handle.write("Model Parameters:\n")
    p = [item for item in temp if not item.startswith('_')]
    for var in sorted(p):
        file_handle.write("{}: {}\n".format(var, getattr(model, var)))

    layers = temp['_modules']  # Returns all top level modules (layers)
    if 'contour_integration_layer' in layers:

        file_handle.write("Contour Integration Layer: {}\n".format(
            model.contour_integration_layer.__class__.__name__))

        # print fixed hyper parameters
        file_handle.write("Hyper parameters\n")

        cont_int_layer_vars = [item for item in vars(model.contour_integration_layer) if
                               not item.startswith('_')]
        for var in sorted(cont_int_layer_vars):
            file_handle.write(
                "\t{}: {}\n".format(var, getattr(model.contour_integration_layer, var)))

        # print parameter names and whether they are trainable
        file_handle.write("Contour Integration Layer Parameters\n")
        layer_params = vars(model.contour_integration_layer)['_parameters']
        for k, v in sorted(layer_params.items()):
            file_handle.write("\t{}: requires_grad {}\n".format(k, v.requires_grad))

    file_handle.write("{}\n".format('-' * 80))
    file_handle.write("Training details\n")
    file_handle.write("Epoch, train_loss, train_iou, val_loss, val_iou, lr\n")

    print("train_batch_size={}, test_batch_size={}, lr={}, epochs={}".format(
        train_batch_size, test_batch_size, learning_rate, num_epochs))

    for epoch in range(num_epochs):

        epoch_start_time = datetime.now()

        train_history.append(train())
        val_history.append(validate())

        lr_history.append(get_lr(optimizer))
        lr_scheduler.step(epoch)

        print(
            "Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, "
            "IoU={:0.4f}. Time {}".format(
                epoch, num_epochs,
                train_history[epoch][0],
                train_history[epoch][1],
                val_history[epoch][0],
                val_history[epoch][1],
                datetime.now() - epoch_start_time
            ))

        if val_history[epoch][1] > best_iou:
            best_iou = val_history[epoch][1]
            torch.save(
                model.state_dict(),
                os.path.join(results_store_dir, 'best_accuracy.pth')
            )

        file_handle.write("[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {}],\n".format(
            epoch,
            train_history[epoch][0],
            train_history[epoch][1],
            val_history[epoch][0],
            val_history[epoch][1],
            lr_history[epoch]
        ))

    training_time = datetime.now() - training_start_time
    print('Finished Training. Training took {}'.format(training_time))

    file_handle.write("{}\n".format('-' * 80))
    file_handle.write("Train Duration       : {}\n".format(training_time))
    file_handle.close()

    # -----------------------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------------------
    train_history = np.array(train_history)
    val_history = np.array(val_history)

    f = plt.figure()
    plt.title("Loss")
    plt.plot(train_history[:, 0], label='train')
    plt.plot(val_history[:, 0], label='validation')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    f.savefig(os.path.join(results_store_dir, 'loss.jpg'), format='jpg')

    f = plt.figure()
    plt.title("IoU")
    plt.plot(train_history[:, 1], label='train')
    plt.plot(val_history[:, 1], label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    f.savefig(os.path.join(results_store_dir, 'iou.jpg'), format='jpg')

    # -----------------------------------------------------------------------------------
    # Run Li 2006 experiments
    # -----------------------------------------------------------------------------------
    print("====> Running Experiments")
    experiment_gain_vs_len.main(model, base_results_dir=results_store_dir, iou_results=False)
    experiment_gain_vs_len.main(
        model, base_results_dir=results_store_dir, iou_results=False,
        frag_size=np.array([11, 11]))
    experiment_gain_vs_spacing.main(model, base_results_dir=results_store_dir)


if __name__ == '__main__':

    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    data_set_parameters = {
        'data_set_dir': './data/bsds',
        # 'train_subset_size': 30000,
        # 'test_subset_size': None
    }

    train_parameters = {
        'train_batch_size': 32,
        'test_batch_size': 1,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'gaussian_reg_weight': 0.0001,
        'gaussian_reg_sigma': 10,
    }

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

    main(net, train_params=train_parameters, data_set_params=data_set_parameters,
         base_results_store_dir='./results/bsds')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
