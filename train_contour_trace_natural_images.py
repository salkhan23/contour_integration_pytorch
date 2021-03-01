# ------------------------------------------------------------------------------------
# Train a contour integration model on Pathfinder in Natural Images Dataset
#
# Ref: https://github.com/xavysp/MBIPED
# Ref: Data: https://drive.google.com/file/d/1l9cUbNK7CgpUsWYInce-djJQp-FyY5DO/view
# Ref: Paper: https://arxiv.org/pdf/1909.01955.pdf
# ------------------------------------------------------------------------------------
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
import dataset_pathfinder
import train_utils
import utils

import experiment_gain_vs_spacing_natural_images


def binary_acc(y_pred, y_target):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_target.float()).sum().float()
    acc = correct_results_sum / y_target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def iterate_epoch(
        model, data_loader, loss_fcn, optimizer1, device, is_train=True, clip_negative_lateral_weights=False):
    """

    :param model:
    :param data_loader:
    :param loss_fcn:
    :param optimizer1:
    :param device:
    :param is_train:
    :param clip_negative_lateral_weights:  clip_negative_lateral_weights: If set will clip negative lateral weights after
       every weight update. Only used when is_train=True

    :return: loss, acc
    """
    if is_train:
        model.train()
    else:
        model.eval()
    torch.set_grad_enabled(is_train)

    e_loss = 0
    e_acc = 0

    for iteration, data_loader_out in enumerate(data_loader, 1):

        if is_train:
            optimizer1.zero_grad()  # zero the parameter gradients

        img, label, _, _, _ = data_loader_out

        img = img.to(device)
        label = label.to(device)

        label_out = model(img)

        total_loss = loss_fcn(
            label_out, label.float(), model.contour_integration_layer.lateral_e.weight,
            model.contour_integration_layer.lateral_i.weight)

        e_loss += total_loss.item()
        acc = binary_acc(label_out, label)
        e_acc += acc.item()

        if is_train:
            total_loss.backward()
            optimizer1.step()

            if clip_negative_lateral_weights:
                model.contour_integration_layer.lateral_e.weight.data = \
                    train_utils.clip_negative_weights(model.contour_integration_layer.lateral_e.weight.data)
                model.contour_integration_layer.lateral_i.weight.data = \
                    train_utils.clip_negative_weights(model.contour_integration_layer.lateral_i.weight.data)

    e_loss = e_loss / len(data_loader)
    e_acc = e_acc / len(data_loader)

    return e_loss, e_acc


def main(model, train_params, data_set_params, cont_int_scale, base_results_store_dir='./results'):

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
    required_training_params = \
        ['train_batch_size', 'test_batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_params:
        assert key in train_params, 'training_params does not have required key {}'.format(key)

    train_batch_size = train_params['train_batch_size']
    test_batch_size = train_params['test_batch_size']
    learning_rate = train_params['learning_rate']
    num_epochs = train_params['num_epochs']

    # Optional
    clip_negative_lateral_weights = train_params.get('clip_negative_lateral_weights', False)

    if 'lr_sched_step_size' not in train_params:
        train_params['lr_sched_step_size'] = 30
    if 'lr_sched_gamma' not in train_params:
        train_params['lr_sched_gamma'] = 0.1
    if 'random_seed' not in train_params:
        train_params['random_seed'] = 1

    torch.manual_seed(train_params['random_seed'])
    np.random.seed(train_params['random_seed'])

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

    # -------------------------------------------------------------------------------
    # Data Loader
    # -------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_load_start_time = datetime.now()

    print("Data Source: {}".format(data_set_dir))

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std),
        utils.PunctureImage(n_bubbles=200, fwhm=np.array([7, 9, 11, 13, 15, 17]))
    ]
    pre_process_transforms = transforms.Compose(transforms_list)

    train_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=os.path.join(data_set_dir, 'train'),
        transform=pre_process_transforms,
    )

    train_batch_size = min(train_batch_size, len(train_set))

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_set = dataset_pathfinder.PathfinderNaturalImages(
        data_dir=os.path.join(data_set_dir, 'test'),
        transform=pre_process_transforms,
    )

    test_batch_size = min(test_batch_size, len(val_set))

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

    # -------------------------------------------------------------------------------
    # Loss / optimizer
    # -------------------------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda params: params.requires_grad, model.parameters()),
        lr=learning_rate
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_params['lr_sched_step_size'],
        gamma=train_params['lr_sched_gamma'])

    # -----------------------------------------------------------------------------------
    # Loss Functions
    # -----------------------------------------------------------------------------------
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = train_utils.ClassBalancedCrossEntropy()
    # criterion = train_utils.ClassBalancedCrossEntropyAttentionLoss()
    criterion_loss_sigmoid_outputs = False

    # Lateral Weights sparsity constraint
    lateral_sparsity_loss = train_utils.InvertedGaussianL1Loss(
        model.contour_integration_layer.lateral_e.weight.shape[2:],
        model.contour_integration_layer.lateral_i.weight.shape[2:],
        train_params['lateral_w_reg_gaussian_sigma'])
    # lateral_sparsity_loss = train_utils.WeightNormLoss(norm=1) # vanilla L1 Loss

    loss_function = train_utils.CombinedLoss(
        criterion=criterion,
        sigmoid_predictions=criterion_loss_sigmoid_outputs,
        sparsity_loss_fcn=lateral_sparsity_loss,
        sparsity_loss_weight=train_params['lateral_w_reg_weight'],
        # negative_weights_loss_fcn=negative_lateral_weights_penalty,
        # negative_weights_loss_weight=negative_lateral_weights_penalty_weight
    ).to(device)

    # -----------------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------------
    print("====> Starting Training ")
    training_start_time = datetime.now()

    train_history = []
    val_history = []
    lr_history = []

    best_acc = 0

    # Summary file
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    file_handle = open(summary_file, 'w+')

    file_handle.write("Data Set Parameters {}\n".format('-' * 60))
    file_handle.write("Source           : {}\n".format(data_set_dir))
    # file_handle.write("Train Set Mean {}, std {}\n".format(
    #     train_set.data_set_mean, train_set.data_set_std))
    # file_handle.write("Validation Set Mean {}, std {}\n".format(
    #     val_set.data_set_mean, train_set.data_set_std))

    file_handle.write("Training Parameters {}\n".format('-' * 60))
    file_handle.write("Random Seed      : {}\n".format(train_params['random_seed']))
    file_handle.write("Train images     : {}\n".format(len(train_set.images)))
    file_handle.write("Val images       : {}\n".format(len(val_set.images)))
    file_handle.write("Train batch size : {}\n".format(train_batch_size))
    file_handle.write("Val batch size   : {}\n".format(test_batch_size))
    file_handle.write("Epochs           : {}\n".format(num_epochs))
    file_handle.write("Optimizer        : {}\n".format(optimizer.__class__.__name__))
    file_handle.write("learning rate    : {}\n".format(learning_rate))
    for key in train_params.keys():
        if 'lr_sched' in key:
            print("  {}: {}".format(key, train_params[key]), file=file_handle)

    file_handle.write("Loss Fcn         : {}\n".format(loss_function.__class__.__name__))
    print(loss_function, file=file_handle)
    file_handle.write("clip negative lateral weights: {}\n".format(clip_negative_lateral_weights))

    file_handle.write("Image pre-processing :\n")
    print(pre_process_transforms, file=file_handle)

    file_handle.write("Model Parameters {}\n".format('-' * 63))
    file_handle.write("Model Name       : {}\n".format(model.__class__.__name__))
    temp = vars(model)  # Returns a dictionary.
    p1 = [item for item in temp if not item.startswith('_')]
    for var in sorted(p1):
        file_handle.write("{}: {}\n".format(var, getattr(model, var)))
    file_handle.write("\n")

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
    file_handle.write("Epoch, train_loss, train_acc, val_loss, val_acc, lr\n")

    print("train_batch_size={}, test_batch_size= {}, lr={}, epochs={}".format(
        train_batch_size, test_batch_size, learning_rate, num_epochs))

    # Track evolution of these variables during training
    # (Must be a parameter of the contour integration layer)
    track_var_dict = {
        'a': [],
        'b': [],
        'j_xy': [],
        'j_yx': [],
        'i_bias': [],
        'e_bias': []
    }

    for epoch in range(0, num_epochs):

        epoch_start_time = datetime.now()

        train_history.append(iterate_epoch(
            model=model,
            data_loader=train_data_loader,
            loss_fcn=loss_function,
            optimizer1=optimizer,
            device=device,
            is_train=True,
            clip_negative_lateral_weights=clip_negative_lateral_weights))

        val_history.append(iterate_epoch(
            model=model,
            data_loader=val_data_loader,
            loss_fcn=loss_function,
            optimizer1=optimizer,
            device=device,
            is_train=False))

        lr_history.append(train_utils.get_lr(optimizer))
        lr_scheduler.step()

        # Track parameters
        cont_int_layer_params = model.contour_integration_layer.state_dict()
        for param in track_var_dict:
            if param in cont_int_layer_params:
                track_var_dict[param].append(cont_int_layer_params[param].cpu().detach().numpy())

        print("Epoch [{}/{}], Train: loss={:0.4f}, Acc={:0.2f}. Val: loss={:0.4f}, Acc={:0.2f}."
              " Time {}".format(
                epoch + 1, num_epochs,
                train_history[epoch][0],
                train_history[epoch][1],
                val_history[epoch][0],
                val_history[epoch][1],
                datetime.now() - epoch_start_time))

        # Save best val accuracy weights
        max_val_acc = val_history[epoch][1]
        if max_val_acc > best_acc:
            best_acc = max_val_acc
            torch.save(
                model.state_dict(),
                os.path.join(results_store_dir, 'best_accuracy.pth')
            )

        # Save Last epoch weights
        torch.save(
            model.state_dict(),
            os.path.join(results_store_dir, 'last_epoch.pth')
        )

        file_handle.write("[{}, {:0.4f}, {:0.2f}, {:0.4f}, {:0.2f}, {}],\n".format(
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
    file_handle.close()

    train_history = np.array(train_history)
    val_history = np.array(val_history)

    train_utils.store_tracked_variables(
        track_var_dict, results_store_dir, n_ch=model.contour_integration_layer.edge_out_ch)

    # -------------------------------------------------------------------
    # PLots
    # -------------------------------------------------------------------
    f, ax_arr = plt.subplots(1, 2)

    ax_arr[0].plot(np.arange(1, num_epochs + 1), train_history[:, 0], label='train')
    ax_arr[0].plot(np.arange(1, num_epochs + 1), val_history[:, 0], label='val')
    ax_arr[0].set_xlabel('Epoch')
    ax_arr[0].set_title("Loss Vs Time")
    ax_arr[0].grid(True)
    ax_arr[0].legend()

    ax_arr[1].plot(np.arange(1, num_epochs + 1), train_history[:, 1], label='train')
    ax_arr[1].plot(np.arange(1, num_epochs + 1), val_history[:, 1], label='val')
    ax_arr[1].set_xlabel('Epoch')
    ax_arr[1].set_title("Accuracy Vs Time")
    ax_arr[1].grid(True)
    ax_arr[1].legend()
    f.savefig(os.path.join(results_store_dir, 'loss_and accuracy.jpg'), format='jpg')

    # -----------------------------------------------------------------------------------
    # Run Li 2006 equivalent experiments
    # -----------------------------------------------------------------------------------
    dataset_parameters = {
        'biped_dataset_dir': './data/BIPED/edges',
        'biped_dataset_type': 'train',
        'n_biped_imgs': 20000,
        'n_epochs': 1  # Total images = n_epochs * n_biped_images
    }

    experiment_gain_vs_spacing_natural_images.main(
        model,
        results_store_dir,
        data_set_params=dataset_parameters,
        cont_int_scale=cont_int_scale
    )

    # View trained kernels
    # ------------------------------------------------------------------------------------
    trained_kernels_store_dir = os.path.join(results_store_dir, 'trained_kernels')
    if not os.path.exists(trained_kernels_store_dir):
        os.makedirs(trained_kernels_store_dir)

    utils.view_ff_kernels(
        model.edge_extract.weight.data.cpu().detach().numpy(),
        results_store_dir=trained_kernels_store_dir
    )

    utils.view_spatial_lateral_kernels(
        model.contour_integration_layer.lateral_e.weight.data.cpu().detach().numpy(),
        model.contour_integration_layer.lateral_i.weight.data.cpu().detach().numpy(),
        results_store_dir=trained_kernels_store_dir,
        spatial_func=np.mean
    )


if __name__ == '__main__':
    plt.ion()

    data_set_parameters = {
        'data_set_dir': './data/pathfinder_natural_images_2',
    }

    train_parameters = {
        'random_seed': 7,
        'train_batch_size': 32,
        'test_batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'lateral_w_reg_weight': 0.0001,
        'lateral_w_reg_gaussian_sigma': 10,
        'clip_negative_lateral_weights': True,
        'lr_sched_step_size': 80,
        'lr_sched_gamma': 0.5
    }

    # # Create Model
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    cont_int_layer = new_piech_models.CurrentDivisiveInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)

    # cont_int_layer = new_control_models.ControlMatchParametersLayer(
    #     lateral_e_size=15, lateral_i_size=15)
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)

    scale_down_input_to_contour_integration_layer = 4
    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)

    main(
        net,
        train_params=train_parameters,
        data_set_params=data_set_parameters,
        base_results_store_dir='./results/contour_tracing',
        cont_int_scale=scale_down_input_to_contour_integration_layer
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print('End')

    import pdb
    pdb.set_trace()
