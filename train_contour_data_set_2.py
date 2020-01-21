# ------------------------------------------------------------------------------------
# Piech - 2013 - Current Based Model with Subtractive Inhibition
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
from models.piech_models import CurrentSubtractiveInhibition, CurrentDivisiveInhibition
from models.new_piech_models import ContourIntegrationCSI
from models.new_control_models import ControlMatchParametersModel
import models.control_models as control_models
import validate_contour_data_set


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
    required_training_params = ['train_batch_size', 'test_batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_params:
        assert key in train_params, 'training_params does not have required key {}'.format(key)
    train_batch_size = train_params['train_batch_size']
    test_batch_size = train_params['test_batch_size']
    learning_rate = train_params['learning_rate']
    num_epochs = train_params['num_epochs']

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Loading Model ")
    print("Name: {}".format(model.__class__.__name__))
    print(model)

    results_store_dir = os.path.join(
        base_results_store_dir,
        model.__class__.__name__ + datetime.now().strftime("_%Y%m%d_%H%M%S"))
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

    # criterion = nn.BCEWithLogitsLoss().to(device)
    loss_fcn = utils.class_balanced_cross_entropy_attention_loss

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

            batch_loss = loss_fcn(torch.sigmoid(label_out), label.float())
            # batch_loss = criterion(label_out, label.float())

            batch_loss.backward()
            optimizer.step()

            e_loss += batch_loss.item()

            preds = (torch.sigmoid(label_out) > detect_thres)
            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

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
                # batch_loss = criterion(label_out, label.float())
                batch_loss = loss_fcn(torch.sigmoid(label_out), label.float())

                e_loss += batch_loss.item()
                preds = (torch.sigmoid(label_out) > detect_thres)
                e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

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
    file_handle.write("Train Set Mean {}, std {}\n".format(train_set.data_set_mean, train_set.data_set_std))
    file_handle.write("Validation Set Mean {}, std {}\n".format(val_set.data_set_mean, train_set.data_set_std))

    file_handle.write("Training Parameters {}\n".format('-' * 60))
    file_handle.write("Train images     : {}\n".format(len(train_set.images)))
    file_handle.write("Val images       : {}\n".format(len(val_set.images)))
    file_handle.write("Train batch size : {}\n".format(train_batch_size))
    file_handle.write("Val batch size   : {}\n".format(test_batch_size))
    file_handle.write("Epochs           : {}\n".format(num_epochs))
    file_handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
    file_handle.write("learning rate    : {}\n".format(learning_rate))
    file_handle.write("Loss Fcn         : {}\n".format(loss_fcn.__name__))
    file_handle.write("IoU Threshold    : {}\n".format(detect_thres))

    file_handle.write("Model Parameters {}\n".format('-' * 63))
    file_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
    file_handle.write("\n")
    print(model, file=file_handle)

    temp = vars(model)  # Returns a dictionary.
    layers = temp['_modules']  # Returns all top level modules (layers)
    if 'contour_integration_layer' in layers:

        # print fixed hyper parameters
        file_handle.write("Contour Integration Layer Hyper parameters\n")
        cont_int_layer_vars = [item for item in vars(model.contour_integration_layer) if not item.startswith('_')]
        for var in sorted(cont_int_layer_vars):
            file_handle.write("\t{}: {}\n".format(var, getattr(model.contour_integration_layer, var)))

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

        print("Epoch [{}/{}], Train: loss={:0.4f}, IoU={:0.4f}. Val: loss={:0.4f}, IoU={:0.4f}. Time {}".format(
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
    plt.legend()
    f.savefig(os.path.join(results_store_dir, 'loss.jpg'), format='jpg')

    f = plt.figure()
    plt.title("IoU")
    plt.plot(train_history[:, 1], label='train')
    plt.plot(val_history[:, 1], label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    f.savefig(os.path.join(results_store_dir, 'iou.jpg'), format='jpg')

    # PLots per Length
    c_len_iou_arr, c_len_loss_arr = \
        validate_contour_data_set.get_performance_per_len(model, data_set_dir, device, c_len_arr)
    f = plt.figure()
    plt.plot(c_len_arr, c_len_iou_arr)
    plt.xlabel("Contour length")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.title("IoU vs Length (Validation Dataset)")
    f.savefig(os.path.join(results_store_dir, 'iou_vs_len.jpg'), format='jpg')
    plt.close(f)

    f = plt.figure()
    plt.plot(c_len_arr, c_len_loss_arr)
    plt.grid()
    plt.xlabel("Contour length")
    plt.ylabel("Loss")
    plt.title("Loss vs Length (Validation Dataset)")
    f.savefig(os.path.join(results_store_dir, 'loss_vs_len.jpg'), format='jpg')
    plt.close(f)


if __name__ == '__main__':

    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    data_set_parameters = {
        'data_set_dir':  "./data/channel_wise_optimal_full14_frag7",
        # 'train_subset_size': 20000,
        # 'test_subset_size': 2000
    }

    train_parameters = {
        'train_batch_size': 16,
        'test_batch_size': 1,
        'learning_rate': 3e-5,
        'num_epochs': 50,
    }

    # net = CurrentDivisiveInhibition().to(device)
    # net = control_models.CmMatchIterations().to(device)
    # net = control_models.CmMatchParameters(lateral_e_size=23, lateral_i_size=23).to(device)
    # net = control_models.CmClassificationHeadOnly().to(device)

    # New
    net = ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # net = ControlMatchParametersModel(lateral_e_size=15, lateral_i_size=15)

    main(net, train_params=train_parameters, data_set_params=data_set_parameters,
         base_results_store_dir='./results/new_model')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
