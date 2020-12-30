# ---------------------------------------------------------------------------------------
# Common Function for Training Scripts
# ---------------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt

import torch

import utils


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def sigmoid(x):
    return 1. / (1+np.exp(-x))


def clip_negative_weights(input_w):
    """ Return input weights with negative values set to zero """
    clipped_w = input_w
    clipped_w[clipped_w < 0] = 0

    return clipped_w


def get_model_layers(m):
    """
    Given a pytorch model/layer returns first layer submodules
    :param m:
    :return:
    """
    temp = vars(m)  # Returns a dictionary.
    layers = temp['_modules']  # Returns all top level modules (layers)
    return layers


# ---------------------------------------------------------------------------------------
# Track/Plots over training
# ---------------------------------------------------------------------------------------

def store_tracked_variables(var_dict, store_dir, n_ch=64):
    """
    Store and Plot variables tracked during Training
    Note: Assumes all tracked variables have n_ch items per epoch.

    :param var_dict: Dictionary of Tracked Variables (key = variable name, value =
        List of tracked values per epoch)
    :param store_dir: base directory. Will Create a subdir "tracked_variables" in
        this directory where summary file & plots will be stored
    :param n_ch: Number of channels of each variable.

    :return: None
    """
    var_requiring_sigmoid = ['a', 'b', 'j_xy', 'j_yx']

    track_v_store_dir = os.path.join(store_dir, 'tracked_variables')
    if not os.path.exists(track_v_store_dir):
        os.makedirs(track_v_store_dir)

    # Store all variables in a file
    track_v_store_file = os.path.join(track_v_store_dir, 'tracked_variables.txt')
    f_handle = open(track_v_store_file, 'w+')

    f_handle.write("Variables Tracked Over Training\n")
    f_handle.write("{}\n".format('-' * 80))

    for key, value in var_dict.items():
        value = np.array(value)
        print("{}:\nnp.{}".format(key, repr(value)), file=f_handle)

    f_handle.close()

    # Plot tracked variables
    single_dim = np.int(np.ceil(np.sqrt(n_ch)))

    for key, value in var_dict.items():
        if not value:
            print("store_tracked_variables: No tracked values for {}".format(key))
            continue

        f1, ax_arr = plt.subplots(single_dim, single_dim, figsize=(9, 9))

        value = np.array(value)

        sigmoid_required = False
        if key in var_requiring_sigmoid:
            sigmoid_required = True

        for ch_idx in range(n_ch):
            r_idx = ch_idx // single_dim
            c_idx = ch_idx - r_idx * single_dim

            if sigmoid_required:
                ax_arr[r_idx, c_idx].plot(sigmoid(value[:, ch_idx]))
            else:
                ax_arr[r_idx, c_idx].plot(value[:, ch_idx])

        if sigmoid_required:
            f1.suptitle("Sigmoid ({})".format(key))
            f1.savefig(os.path.join(track_v_store_dir, 'sigma_{}.jpg'.format(key)), format='jpg')
        else:
            f1.suptitle("{}".format(key))
            f1.savefig(os.path.join(track_v_store_dir, '{}.jpg'.format(key)), format='jpg')

        plt.close()


def plot_training_history(train_history, val_history, store_dir=None):
    """
    Plot loss and IoU scores over Epoch
    :param train_history: Array of [loss, iou] of length n_epochs
    :param val_history: Array of [loss, iou] of length n_epochs
    :param store_dir: directory where to store results
    :return:
    """
    f = plt.figure()
    plt.title("Loss")
    plt.plot(train_history[:, 0], label='train')
    plt.plot(val_history[:, 0], label='validation')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    if store_dir is not None:
        f.savefig(os.path.join(store_dir, 'loss_per_epoch.jpg'), format='jpg')
        plt.close(f)

    f = plt.figure()
    plt.title("IoU")
    plt.plot(train_history[:, 1], label='train')
    plt.plot(val_history[:, 1], label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    if store_dir is not None:
        f.savefig(os.path.join(store_dir, 'iou_per_epoch.jpg'), format='jpg')
        plt.close(f)


# ---------------------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------------------
class InvertedGaussianL1Loss(torch.nn.Module):
    def __init__(self, e_mask_size, i_mask_size, mask_width):
        """
        L1 weight regularization loss. An inverted gaussian mask is applied over the
        weights and the absolute L1 loss is calculated. Weights far away from the center
        are penalized more compared to those close to the center.

        :param e_mask_size:
        :param i_mask_size:
        :param mask_width: sigma of the Gaussian mask
        """
        super(InvertedGaussianL1Loss, self).__init__()

        self.e_mask_size = e_mask_size
        self.i_mask_size = i_mask_size
        self.mask_width = mask_width

        # Register masks as buffers, so they can move to the device when to function is called.
        # self.mask_e = 1 - utils.get_2d_gaussian_kernel(e_mask_size, mask_width)
        # self.mask_i = 1 - utils.get_2d_gaussian_kernel(i_mask_size, mask_width)
        # self.mask_e = torch.from_numpy(self.mask_e).float()
        # self.mask_i = torch.from_numpy(self.mask_i).float()
        self.register_buffer(
            'mask_e', torch.from_numpy(1 - utils.get_2d_gaussian_kernel(e_mask_size, mask_width)))
        self.register_buffer(
            'mask_i', torch.from_numpy(1 - utils.get_2d_gaussian_kernel(e_mask_size, mask_width)))

    def forward(self, weight_e, weight_i):
        loss1 = (self.mask_e * weight_e).abs().sum() + (self.mask_i * weight_i).abs().sum()
        return loss1

    def __repr__(self):
        string = "InvertedGaussianL1Loss()\n"
        string += ("  e_mask_size         : {}\n".format(self.e_mask_size))
        string += ("  i_mask_size         : {}\n".format(self.i_mask_size))
        string += ("  Gaussian mask sigma : {}".format(self.mask_width))
        return string


class WeightNormLoss(torch.nn.Module):
    def __init__(self, norm=1):
        """
        Regular Weight Loss. L1/L2 loss as specified by the norm parameter

        :param norm: [default is 1 (L1 loss)]
        """
        super(WeightNormLoss, self).__init__()
        self.norm = norm

    def forward(self, weight_e, weight_i):
        loss = \
            weight_e.norm(p=self.norm) + \
            weight_i.norm(p=self.norm)
        return loss

    def __repr__(self):
        string = "WeightNormLoss()\n"
        string += ("  Norm : {}".format(self.norm))
        return string


def negative_weights_loss(weight_e, weight_i):
    """
    Find the negative weights in the two supplied weights, square them and sum them to get loss

    :param weight_e:
    :param weight_i:
    :return:
    """
    neg_weight_e = weight_e[weight_e < 0]
    neg_weight_i = weight_i[weight_i < 0]

    loss1 = torch.pow(neg_weight_e, 2).sum() + torch.pow(neg_weight_i, 2).sum()

    return loss1


class BceAndLateralWeightSparsityLoss(torch.nn.Module):
    def __init__(self, sparsity_loss_fcn, sparsity_loss_weight):
        """
        BCEWithLogitsLoss() and  specified sparsity loss collectively

        :param sparsity_loss_fcn:
        :param sparsity_loss_weight:
        """
        super(BceAndLateralWeightSparsityLoss, self).__init__()

        self.sparsity_loss = sparsity_loss_fcn
        self.sparsity_loss_weight = sparsity_loss_weight

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, label_out, label, weight_e, weight_i):

        criteria_loss = self.criterion(label_out, label)
        w_sparsity_loss = self.sparsity_loss_weight * self.sparsity_loss(weight_e, weight_i)

        total_loss = criteria_loss + w_sparsity_loss

        # print("Total Loss {:0.4f}. Criteria Loss {:0.4f}, Weight Sparsity Loss {:0.4f}".format(
        #     total_loss, criteria_loss, w_sparsity_loss))

        return total_loss

    def __repr__(self):
        string = "BceAndLateralWeightSparsityLoss()\n"
        string += "  Criterion Loss       : {}\n".format(self.criterion)
        string += ("  Sparsity Loss        : {}\n".format(self.sparsity_loss))
        string += ("  Sparsity loss weight : {}".format(self.sparsity_loss_weight))
        return string


class BceAndLateralWeightSparsityAndNegativeWeightPenalty(torch.nn.Module):
    def __init__(
            self, sparsity_loss_fcn, sparsity_loss_weight,
            negative_weights_penalty_fcn, negative_weights_penalty_weight):
        """

         BCEWithLogitsLoss(), specified sparsity loss collectively and specified negative weight loss

        :param sparsity_loss_fcn:
        :param sparsity_loss_weight:
        :param negative_weights_penalty_fcn:
        :param negative_weights_penalty_weight:
        """
        super(BceAndLateralWeightSparsityAndNegativeWeightPenalty, self).__init__()

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.sparsity_loss_fcn = sparsity_loss_fcn
        self.sparsity_loss_weight = sparsity_loss_weight

        self.negative_weights_penalty_fcn = negative_weights_penalty_fcn
        self.negative_weights_penalty_weight = negative_weights_penalty_weight

    def forward(self, label_out, label, weight_e, weight_i):

        criteria_loss = self.criterion(label_out, label)
        w_sparsity_loss = self.sparsity_loss_weight * self.sparsity_loss_fcn(weight_e, weight_i)
        neg_weights_penalty = \
            self.negative_weights_penalty_weight * self.negative_weights_penalty_fcn(weight_e, weight_i)

        total_loss = criteria_loss + w_sparsity_loss + neg_weights_penalty

        # print("Total Loss {:0.4f}. Criteria Loss {:0.4f}, Weight Sparsity Loss {:0.4f}. Negative Weights Loss {:0.4f}".format(
        #     total_loss, criteria_loss, w_sparsity_loss, neg_weights_penalty))

        return total_loss

    def __repr__(self):
        string = "BceAndLateralWeightSparsityAndNegativeWeightPenalty()\n"
        string += "  Criterion Loss           : {}\n".format(self.criterion)
        string += ("  Sparsity Loss            : {}\n".format(self.sparsity_loss_fcn))
        string += ("  Sparsity loss weight     : {}\n".format(self.sparsity_loss_weight))
        string += ("  Negative w. Loss         : {}\n".format(self.negative_weights_penalty_fcn.__name__))
        string += ("  Negative w. Loss weight  : {}\n".format(self.negative_weights_penalty_weight))
        return string
