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


# ***************************************************************************************
#                                 Loss Functions
# ***************************************************************************************
# ---------------------------------------------------------------------------------------
#  Criterion Loss Functions
# ---------------------------------------------------------------------------------------
class ClassBalancedCrossEntropy(torch.nn.Module):
    def __init__(self):
        """
        Dynamically find the number of edges and non-edges and scale edge and non-edge losses.
        NOTE: This function requires sigmoided outputs

        [1] In an image there are a lot more non-boundary pixels compared with boundary pixels.
        To account for this, a dynamic parameters (alpha) is used to adjust the losses for
        each class.

        REF: originally proposed in
        [Xie, S., Tu, Z.:  Holistically-nested edge detection.
        In: Proceedings of the IEEE international conference on computer vision. (2015) 1395–1403]

        Used Reference:
        [Wang, Liang and Li -2018- DOOBNet: Deep Object Occlusion BoundaryDetection from an Image
        """
        super(ClassBalancedCrossEntropy, self).__init__()

    def forward(self, outputs, targets):
        # clamp extrema, the log does not like 0 values
        outputs = torch.clamp(outputs, 1e-6, 1 - 1e-6)

        n_total = targets.shape[0] * targets.shape[1] * targets.shape[2] * targets.shape[3]
        n_non_contours = torch.nonzero(targets).shape[0]
        # n_contours = n_total - n_non_contours

        alpha = n_non_contours / n_total

        # print("Batch: Num Fragments={}, Num contour={}, num non-contour={}. alpha = {}".format(
        #      n_total, n_contours, n_non_contours, alpha))

        contour_loss = -targets * alpha * torch.log(outputs)
        non_contour_loss = (targets - 1) * (1 - alpha) * torch.log(1 - outputs)

        loss = torch.sum(contour_loss + non_contour_loss)
        loss = loss / outputs.shape[0]  # batch size

        return loss

    def __repr__(self):
        string = "ClassBalancedCrossEntropy()\n"
        return string


class ClassBalancedCrossEntropyAttentionLoss(torch.nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        """
        Doobnet Loss.
        Here boundary = contour fragment and non-boundary = non-contour fragment.
        NOTE: This function requires sigmoided outputs

        [1] In an image there are a lot more non-boundary pixels compared with boundary pixels.
        To account for this, a dynamic parameters (alpha) is used to adjust the losses for each class.

        [2] The class based cross entropy is then weighted with two parameters, beta and gamma. This has
        the effect of putting more emphasis (higher loss) for mis-identifications . False positives
        and False negatives.  Refer to Doobnet paper for more details.

        Compared to Doobnet repository this is the attentional loss without sigmoid (similar to paper)
        In the repository, the final network sigmoid is included with this loss.

        :param beta:
        :param gamma:
        """
        super(ClassBalancedCrossEntropyAttentionLoss, self).__init__()

        self.gamma = gamma
        self.beta = beta

    def forward(self, outputs, targets):
        # Calculate alpha (class balancing weight).
        # -----------------------------------------
        # Way fewer boundary pixels compared with non-boundary pixels
        num_total = targets.shape[0] * targets.shape[1] * targets.shape[2] * targets.shape[3]
        num_boundary = (torch.nonzero(targets)).shape[0]
        num_non_boundary = num_total - num_boundary

        alpha = num_non_boundary / num_total

        # print("Batch: Num Pixels={}, Num boundary={}, num non-boundary={}. alpha = {}".format(
        #     num_total, num_boundary, num_non_boundary, alpha))

        # calculate loss
        # ---------------
        # Fast Way
        # ---------------
        # clamp extrema, the log does not like 0 values
        outputs = torch.clamp(outputs, 1e-6, 1 - 1e-6)

        boundary_loss = targets * -alpha * (self.beta ** ((1 - outputs) ** self.gamma)) * torch.log(outputs)
        non_boundary_loss = (targets - 1) * (1 - alpha) * (self.beta ** (outputs ** self.gamma)) * torch.log(
            1 - outputs)

        loss = torch.sum(boundary_loss + non_boundary_loss)
        loss = loss / outputs.shape[0]  # batch size

        # ------------
        # Slow Way
        # ------------
        # flat_outputs = torch.reshape(outputs, (num_total, 1))
        # flat_targets = torch.reshape(targets, (num_total, 1))
        #
        # loss = 0
        # for p_idx in range(num_total):
        #
        #     if flat_targets[p_idx] == 1:  # is a boundary
        #         loss += -alpha * (beta**(1 - flat_outputs[p_idx])**gamma) * torch.log(flat_outputs[p_idx])
        #     else:
        #         loss += -(1 - alpha) * beta**((flat_outputs[p_idx]**gamma)) * torch.log(1 - flat_outputs[p_idx])
        #     # print("[{}/{}] AT Loss={}".format(loss.data[0], p_idx, num_total))

        # print("AT Loss={}".format(loss.data[0]))
        z = loss.cpu().detach().numpy()
        if np.isnan(z):
            print("*" * 80)
            print("Loss is NaN")
            import pdb
            pdb.set_trace()

        return loss

    def __repr__(self):
        string = "ClassBalancedCrossEntropyAttentionLoss()\n"
        string += ("  beta          : {}\n".format(self.beta))
        string += ("  gamma         : {}\n".format(self.gamma))
        return string


# ---------------------------------------------------------------------------------------
#  Weight Sparsity Loss Functions
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


# ---------------------------------------------------------------------------------------
#  Negative Weight Loss Functions
# ---------------------------------------------------------------------------------------
class NegativeWeightsNormLoss(torch.nn.Module):
    def __init__(self, norm=2):
        """

        :param norm: [Default = 2 (L2 Loss)]
        """
        super(NegativeWeightsNormLoss, self).__init__()
        self.norm = norm

    def forward(self, weight_e, weight_i):

        neg_weight_e = weight_e[weight_e < 0]
        neg_weight_i = weight_i[weight_i < 0]

        loss = \
            neg_weight_e.norm(p=self.norm) + \
            neg_weight_i.norm(p=self.norm)

        return loss

    def __repr__(self):
        string = "NegativeWeightsNormLoss()\n"
        string += ("  Norm : {}".format(self.norm))
        return string


# ---------------------------------------------------------------------------------------
#  Combined Losses Function
# ---------------------------------------------------------------------------------------
class CombinedLoss(torch.nn.Module):
    def __init__(
            self, criterion=torch.nn.BCEWithLogitsLoss, sigmoid_predictions=False,
            sparsity_loss_fcn=None, sparsity_loss_weight=0, negative_weights_loss_fcn=None,
            negative_weights_loss_weight=0):

        super(CombinedLoss, self).__init__()

        if isinstance(criterion, (ClassBalancedCrossEntropy, ClassBalancedCrossEntropyAttentionLoss)):
            if not sigmoid_predictions:
                raise Exception("{} requires sigmoided outputs but sigmoid_predictions is False".format(
                    type(criterion)))

        if sparsity_loss_fcn is not None:
            if sparsity_loss_weight == 0:
                raise Exception("Sparsity Loss function specified without providing loss weight")

        if negative_weights_loss_fcn is not None:
            if negative_weights_loss_weight == 0:
                raise Exception("Negative weights loss function specified without providing loss weight")

        self.criterion = criterion
        self.sigmoid_predictions = sigmoid_predictions
        self.sparsity_loss_fcn = sparsity_loss_fcn
        self.sparsity_loss_weight = sparsity_loss_weight
        self.negative_weights_loss_fcn = negative_weights_loss_fcn
        self.negative_weights_loss_weight = negative_weights_loss_weight

    def forward(self, label_out, label, weight_e=None, weight_i=None):

        if self.sigmoid_predictions:
            label_out = torch.sigmoid(label_out)

        criteria_loss = self.criterion(label_out, label)

        sparsity_loss = 0
        if self.sparsity_loss_fcn is not None:
            sparsity_loss = self.sparsity_loss_weight * self.sparsity_loss_fcn(weight_e, weight_i)

        neg_weights_penalty = 0
        if self.negative_weights_loss_fcn is not None:
            neg_weights_penalty = \
                self.negative_weights_loss_weight * self.negative_weights_loss_fcn(weight_e, weight_i)

        total_loss = criteria_loss + sparsity_loss + neg_weights_penalty

        # print("Total Loss {:0.4f}. Criteria {:0.4f}, Lateral W. Sparsity {:0.4f}. "
        #       "Negative Lateral W. {:0.4f}".format(
        #         total_loss, criteria_loss, sparsity_loss, neg_weights_penalty))

        return total_loss

    def __repr__(self):
        string = "CombinedLoss()\n"
        string += "  Criterion Loss           : {}\n".format(self.criterion)
        string += "    Sigmoid output for criterion loss : {}\n".format(self.sigmoid_predictions)
        if self.sparsity_loss_fcn:
            string += ("  Sparsity Loss            : {}\n".format(self.sparsity_loss_fcn))
            string += ("  Sparsity loss weight     : {}\n".format(self.sparsity_loss_weight))

        if self.negative_weights_loss_fcn:
            string += ("  Negative w. Loss         : {}\n".format(self.negative_weights_loss_fcn))
            string += ("  Negative w. Loss weight  : {}\n".format(self.negative_weights_loss_weight))

        return string
