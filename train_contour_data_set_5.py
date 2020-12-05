# ------------------------------------------------------------------------------------
# Similar to train_contour_data_set_3.py but with additional constraint of only
# positive lateral weights. Currently this is enforced by setting all negative weights to zero
# after every epoch
#
# Similar to original script but with gaussian weight regularization on the lateral
# connections.
# ------------------------------------------------------------------------------------
from datetime import datetime
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset
import utils
import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import validate_contour_data_set

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

    # Optional
    train_subset_size = data_set_params.get('train_subset_size', None)
    test_subset_size = data_set_params.get('test_subset_size', None)
    c_len_arr = data_set_params.get('c_len_arr', None)
    beta_arr = data_set_params.get('beta_arr', None)
    alpha_arr = data_set_params.get('alpha_arr', None)
    gabor_set_arr = data_set_params.get('gabor_set_arr', None)

    # Validate training parameters
    # ----------------------------
    required_training_params = \
        ['train_batch_size', 'test_batch_size', 'learning_rate', 'num_epochs']
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
    print("Restrictions:\n clen={},\n beta={},\n alpha={},\n gabor_sets={},\n train_subset={},\n "
          "test subset={}\n".format(
            c_len_arr, beta_arr, alpha_arr, gabor_set_arr, train_subset_size, test_subset_size))

    # get mean/std of dataset
    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as file_handle:
        meta_data = pickle.load(file_handle)
    # print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    # Pre-processing
    normalize = transforms.Normalize(
        mean=meta_data['channel_mean'],
        std=meta_data['channel_std']
    )

    train_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'train'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize,
        subset_size=train_subset_size,
        c_len_arr=c_len_arr,
        beta_arr=beta_arr,
        alpha_arr=alpha_arr,
        gabor_set_arr=gabor_set_arr
    )

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_set = dataset.Fields1993(
        data_dir=os.path.join(data_set_dir, 'val'),
        bg_tile_size=meta_data["full_tile_size"],
        transform=normalize,
        subset_size=test_subset_size,
        c_len_arr=c_len_arr,
        beta_arr=beta_arr,
        alpha_arr=alpha_arr,
        gabor_set_arr=gabor_set_arr
    )

    val_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=True,
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
        filter(lambda p: p.requires_grad, model.parameters()),
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

    # lambda1 = 0.0001

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
            #
            # import pdb
            # pdb.set_trace()

            total_loss.backward()
            optimizer.step()

            e_loss += total_loss.item()

            preds = (torch.sigmoid(label_out) > detect_thres)
            e_iou += utils.intersection_over_union(
                preds.float(), label.float()).cpu().detach().numpy()

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
                e_iou += utils.intersection_over_union(
                    preds.float(), label.float()).cpu().detach().numpy()

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
    file_handle.write("Restrictions     :\n")
    file_handle.write("  Lengths        : {}\n".format(c_len_arr))
    file_handle.write("  Beta           : {}\n".format(beta_arr))
    file_handle.write("  Alpha          : {}\n".format(alpha_arr))
    file_handle.write("  Gabor Sets     : {}\n".format(gabor_set_arr))
    file_handle.write("  Train Set Size : {}\n".format(train_subset_size))
    file_handle.write("  Test Set Size  : {}\n".format(test_subset_size))
    file_handle.write("Train Set Mean {}, std {}\n".format(
        train_set.data_set_mean, train_set.data_set_std))
    file_handle.write("Validation Set Mean {}, std {}\n".format(
        val_set.data_set_mean, train_set.data_set_std))

    file_handle.write("Training Parameters {}\n".format('-' * 60))
    file_handle.write("Random Seed      : {}\n".format(random_seed))
    file_handle.write("Train images     : {}\n".format(len(train_set.images)))
    file_handle.write("Val images       : {}\n".format(len(val_set.images)))
    file_handle.write("Train batch size : {}\n".format(train_batch_size))
    file_handle.write("Val batch size   : {}\n".format(test_batch_size))
    file_handle.write("Epochs           : {}\n".format(num_epochs))
    file_handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
    file_handle.write("learning rate    : {}\n".format(learning_rate))
    file_handle.write("Loss Fcn         : {}\n".format(criterion.__class__.__name__))
    file_handle.write("Gaussian Regularization sigma        : {}\n".format(gaussian_kernel_sigma))
    file_handle.write("Gaussian Regularization weight        : {}\n".format(lambda1))
    file_handle.write("IoU Threshold    : {}\n".format(detect_thres))

    file_handle.write("Model Parameters {}\n".format('-' * 63))
    file_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
    p1 = [item for item in temp if not item.startswith('_')]
    for var in sorted(p1):
        file_handle.write("{}: {}\n".format(var, getattr(model, var)))
    file_handle.write("\n")
    print(model, file=file_handle)

    temp = vars(model)  # Returns a dictionary.
    layers = temp['_modules']  # Returns all top level modules (layers)
    if 'contour_integration_layer' in layers:

        # print fixed hyper parameters
        file_handle.write("Contour Integration Layer:\n")
        file_handle.write("Type : {}\n".format(model.contour_integration_layer.__class__.__name__))
        cont_int_layer_vars = \
            [item for item in vars(model.contour_integration_layer) if not item.startswith('_')]
        for var in sorted(cont_int_layer_vars):
            file_handle.write("\t{}: {}\n".format(
                var, getattr(model.contour_integration_layer, var)))

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

    a_track_list = []
    b_track_list = []

    j_xy_track_list = []
    j_yx_track_list = []

    i_bias_track_list = []
    e_bias_track_list = []

    for epoch in range(0, num_epochs):

        epoch_start_time = datetime.now()

        train_history.append(train())
        val_history.append(validate())

        lr_history.append(get_lr(optimizer))
        lr_scheduler.step(epoch)

        a_track_list.append(model.contour_integration_layer.a.detach().cpu().numpy())
        b_track_list.append(model.contour_integration_layer.b.detach().cpu().numpy())
        j_xy_track_list.append(model.contour_integration_layer.j_xy.detach().cpu().numpy())
        j_yx_track_list.append(model.contour_integration_layer.j_yx.detach().cpu().numpy())
        i_bias_track_list.append(model.contour_integration_layer.i_bias.detach().cpu().numpy())
        e_bias_track_list.append(model.contour_integration_layer.e_bias.detach().cpu().numpy())

        # Clip all negative lateral weights
        # ----------------------------------------------------------------------------------
        positive_weights = model.contour_integration_layer.lateral_e.weight.data
        positive_weights[positive_weights < 0] = 0
        model.contour_integration_layer.lateral_e.weight.data = positive_weights

        positive_weights = model.contour_integration_layer.lateral_i.weight.data
        positive_weights[positive_weights < 0] = 0
        model.contour_integration_layer.lateral_i.weight.data = positive_weights

        print("Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, IoU={:0.4f}. "
              "Time {}".format(
                epoch + 1, num_epochs,
                train_history[epoch][0],
                train_history[epoch][1],
                val_history[epoch][0],
                val_history[epoch][1],
                datetime.now() - epoch_start_time))

        if val_history[epoch][1] > best_iou:
            best_iou = val_history[epoch][1]
            torch.save(
                model.state_dict(),
                os.path.join(results_store_dir, 'best_accuracy.pth')
            )

        file_handle.write("[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {}],\n".format(
            epoch + 1,
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

    # ----------------------------------------------------------------------------------
    # Track some variables across training
    # ----------------------------------------------------------------------------------
    def sigmoid(x):
        return 1. / (1+np.exp(-x))

    np.set_printoptions(precision=3, linewidth=100, suppress=True, threshold=np.inf)
    file_handle.write("{}\n".format('-' * 80))

    a_track_list = np.array(a_track_list)
    print("a : \n" + 'np.' + repr(a_track_list), file=file_handle)

    b_track_list = np.array(b_track_list)
    print("b : \n" + 'np.' + repr(b_track_list), file=file_handle)

    j_xy_track_list = np.array(j_xy_track_list)
    print("Jxy : \n" + 'np.' + repr(j_xy_track_list), file=file_handle)

    j_yx_track_list = np.array(j_yx_track_list)
    print("Jyx : \n" + 'np.' + repr(j_yx_track_list), file=file_handle)

    i_bias_track_list = np.array(i_bias_track_list)
    print("i_bias : \n" + 'np.' + repr(i_bias_track_list), file=file_handle)

    e_bias_track_list = np.array(e_bias_track_list)
    print("e_bias : \n" + 'np.' + repr(e_bias_track_list), file=file_handle)

    # plot Tracked Variables
    # Sigma(a)
    # ---------
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(sigmoid(a_track_list[:, ch_idx]))
        ax_arr[r_idx, c_idx].set_ylim([0, 1])
    f1.suptitle("Sigmoid (a)")
    f1.savefig(os.path.join(results_store_dir, 'sigma_a.jpg'), format='jpg')
    plt.close(f1)

    # Sigma(b)
    # ---------
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(sigmoid(b_track_list[:, ch_idx]))
        ax_arr[r_idx, c_idx].set_ylim([0, 1])
    f1.suptitle("Sigmoid (b)")
    f1.savefig(os.path.join(results_store_dir, 'sigma_b.jpg'), format='jpg')
    plt.close(f1)

    # J_xy
    # ----
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(j_xy_track_list[:, ch_idx])
        ax_arr[r_idx, c_idx].set_limit([-1, 1])
    f1.suptitle("J_xy")
    f1.savefig(os.path.join(results_store_dir, 'j_xy.jpg'), format='jpg')
    plt.close(f1)

    # J_yx
    # ----
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(j_yx_track_list[:, ch_idx])
        ax_arr[r_idx, c_idx].set_limit([-1, 1])
    f1.suptitle("J_yx")
    f1.savefig(os.path.join(results_store_dir, 'j_yx.jpg'), format='jpg')
    plt.close(f1)

    # i_bias
    # ------
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(i_bias_track_list[:, ch_idx])
        ax_arr[r_idx, c_idx].set_limit([-1, 1])
    f1.suptitle("i_bias")
    f1.savefig(os.path.join(results_store_dir, 'i_bias.jpg'), format='jpg')
    plt.close(f1)

    # e_bias
    # ------
    f1, ax_arr = plt.subplots(8, 8, figsize=(9, 9))
    for ch_idx in range(64):
        r_idx = ch_idx // 8
        c_idx = ch_idx - r_idx * 8
        ax_arr[r_idx, c_idx].plot(e_bias_track_list[:, ch_idx])
        ax_arr[r_idx, c_idx].set_limit([-1, 1])
    f1.suptitle("e_bias")
    f1.savefig(os.path.join(results_store_dir, 'e_bias.jpg'), format='jpg')
    plt.close(f1)

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

    # PLots per Length
    c_len_arr = [1, 3, 5, 7, 9]

    c_len_iou_arr, c_len_loss_arr = validate_contour_data_set.get_performance_per_len(
        model, data_set_dir, device, beta_arr=[0], c_len_arr=c_len_arr)
    f = plt.figure()
    plt.plot(c_len_arr, c_len_iou_arr)
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.axhline(np.mean(c_len_iou_arr), label='average_iou = {:0.2f}'.format(
        np.mean(c_len_iou_arr)), color='red', linestyle=':')
    plt.legend()
    plt.title("IoU vs Length (Validation Dataset-Straight Contours)")
    f.savefig(os.path.join(results_store_dir, 'iou_vs_len.jpg'), format='jpg')
    plt.close(f)

    file_handle.write("{}\n".format('-' * 80))
    file_handle.write("Contour Length vs IoU (Validation) : {}\n".format(c_len_iou_arr))

    # f = plt.figure()
    # plt.plot(c_len_arr, c_len_loss_arr)
    # plt.grid()
    # plt.xlabel("Contour length")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Length (Validation Dataset)-Straight Contours")
    # f.savefig(os.path.join(results_store_dir, 'loss_vs_len.jpg'), format='jpg')
    # plt.close(f)

    # # -----------------------------------------------------------------------------------
    # # Run Li 2006 experiments
    # # -----------------------------------------------------------------------------------
    # print("====> Running Experiments")
    # experiment_gain_vs_len.main(model, base_results_dir=results_store_dir)
    # experiment_gain_vs_spacing.main(model, base_results_dir=results_store_dir)

    file_handle.close()


if __name__ == '__main__':

    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    data_set_parameters = {
        'data_set_dir':  "./data/channel_wise_optimal_full14_frag7",
        'train_subset_size': 20000,
        'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 32,
        'test_batch_size': 1,
        'learning_rate': 3e-5,
        'num_epochs': 4,
        'gaussian_reg_weight': 0.0001,
        'gaussian_reg_sigma': 10,
    }

    # Build Model
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # cont_int_layer = new_control_models.ControlMatchParametersLayer(
    #     lateral_e_size=15, lateral_i_size=15)
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)

    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

    # net = ControlMatchParametersModel(lateral_e_size=15, lateral_i_size=15)

    main(net, train_params=train_parameters, data_set_params=data_set_parameters,
         base_results_store_dir='./results/positive_weights')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
