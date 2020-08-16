# ---------------------------------------------------------------------------------------
# Contour Gain vs fragment spacing variant for natural images.
# Specific placement of occlusion bubbles are used to generate fragmented contours.
#
# Similar to gain_vs_spacing_natural_images.py but uses a different definition for gain.
# Contour Integration Gain = output act various RCD / output act RCD = 1
#
# RCD = ratio of spacing (bubble length) vs fragment length
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import heapq

import torch
from torchvision import transforms

import models.new_piech_models as new_piech_models
import models.new_control_models as new_control_models
from generate_pathfinder_dataset import OnlineNaturalImagesPathfinder
from torch.utils.data import DataLoader
import utils

edge_extract_act = []
cont_int_in_act = []
cont_int_out_act = []


def disable_print():
    """ Disable printing """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """ Enable print """
    sys.stdout = sys.__stdout__


def edge_extract_cb(self, layer_in, layer_out):
    """
    Callback to Retrieve the activations output of edge Extract layer
    Attach at Edge Extract layer
    """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """
    Callback to Retrieve the activations input & output of the contour Integration layer
    Attach at Contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


def process_image(model, devise_to_use, ch_mus, ch_sigmas, in_img):
    """
    Pass image through model and get sigmoid prediction of the model

    :param model:
    :param devise_to_use:
    :param in_img:
    :param ch_mus:
    :param ch_sigmas:

    :return: sigmoided output of the model
    """
    # Zero all collected variables
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    edge_extract_act = 0
    cont_int_in_act = 0
    cont_int_out_act = 0

    normalize = transforms.Normalize(mean=ch_mus, std=ch_sigmas)
    model_in_img = normalize(in_img.squeeze())
    model_in_img = model_in_img.to(devise_to_use).unsqueeze(0)

    # Pass the image through the model
    model.eval()
    if isinstance(model, new_piech_models.JointPathfinderContourResnet50):
        # Output is contour_dataset_out, pathfinder_out
        _, label_out = model(model_in_img)
    else:
        label_out = model(model_in_img)

    label_out = torch.sigmoid(label_out)

    return label_out


class MaxActiveElement:
    """ What to track for each image when finding optimal stimuli for channels """
    def __init__(self, in_act, out_act, position, index, c1, c2, ep1, ep2, prediction, gt):
        self.in_act = in_act
        self.out_act = out_act
        self.position = position
        self.index = index  # index of the original image
        self.c1 = c1  # always the connected contour
        self.c2 = c2
        self.ep1 = ep1
        self.ep2 = ep2
        self.prediction = prediction
        self.gt = gt

    def print(self):
        print("Out act {:0.4f}. corresponding In Act = {:0.4f} position {}, "
              "original image idx {}. Prediction {:0.4f}".format(
                self.out_act, self.in_act, self.position, self.index, self.prediction))


class TopNTracker(object):
    """
    A Priority Queue to track an ordered list of top-n images that a channel responded
    maximally when parsing a dataset.

    priority que automatically arranges items in ascending order. Popped items are lowest values.

    :param: depth = top-n (number of images to track)
    """

    def __init__(self, depth=5):
        self._heap = []
        self.depth = depth

    def push(self, value, count, item):
        """
        The count variable is added to make items unique in case max activations are equal.
        In this case, heappq will try to compare the next item in the tuple (MaxActiveElement)
        which it does not know how to compare. The count must be unique

        REF: https://stackoverflow.com/questions/42985030/inserting-dictionary-to-heap-python
        """

        if len(self._heap) < self.depth:
            heapq.heappush(self._heap, (value, count, item))
        else:
            min_stored_val, _, min_stored_item = self.pop()  # pop the lowest

            if value > min_stored_val:
                heapq.heappush(self._heap, (value, count, item))

    def pop(self):
        return heapq.heappop(self._heap)

    def __len__(self):
        return len(self._heap)

    def print(self):
        print(self._heap)

    def get_stored_values(self):
        lst_items = []
        lst_values = []

        while len(self._heap) > 0:

            v, count, item = self.pop()
            lst_items.append(item)
            lst_values.append(v)

        if len(lst_items):
            idxs = np.argsort(lst_values)
            idxs = idxs[::-1]  # reverse the order, max first.

            lst_items = [lst_items[idx] for idx in idxs]
            lst_values = [lst_values[idx] for idx in idxs]

        return lst_items, lst_values


def get_closest_distance_and_index(desired_d, d_arr):
    """ Find the closest element of d_arr to desired_d """
    offset_dist_arr = np.abs(d_arr - desired_d)
    min_dist = np.min(offset_dist_arr)
    idx = np.argmin(offset_dist_arr)

    return min_dist, idx


def _rhs_get_bubbles_locations(c1, frag_len, bubble_len, ds):
    """ For use see get_bubbles_locations"""
    bubbles_locations_arr = []
    n_bubbles = 0
    while len(c1) > 1:
        d_arr = ds.get_distance_point_and_contour(c1[0], c1)

        desired_d = bubble_len + frag_len
        if 0 == n_bubbles:
            desired_d = desired_d // 2  # first bubble location is different

        min_dist, idx = get_closest_distance_and_index(desired_d, d_arr)
        if min_dist < 1:
            # print("Adding point {} at index {} to bubble_loc array".format(c1[idx], idx))
            bubbles_locations_arr.append(c1[idx])

        n_bubbles += 1
        c1 = c1[idx:]

    return bubbles_locations_arr


def get_bubbles_locations(contour, start_point_idx, frag_len, bubble_len, ds):
    """
    Starting at the start point (specified by index in c1) iteratively parse contour c1
    to find insert location of bubbles that result in a contour with fragments of size frag_len
    and separated by bubble_len

    The first visible fragment is centered at the starting point

    ds = link to pathfinder dataset . Used to access some internal functions.

    NOTE: This returns the CENTER location of the bubble

    """
    # RHS
    c1 = contour[start_point_idx:]
    bubble_locations = _rhs_get_bubbles_locations(c1, frag_len, bubble_len, ds)

    # LHS
    c1 = contour[:start_point_idx]
    c1 = c1[::-1]  # reverse it and use the same way
    lhs_bubbles = _rhs_get_bubbles_locations(c1, frag_len, bubble_len, ds)
    bubble_locations.extend(lhs_bubbles)

    bubble_locations = np.array(bubble_locations)

    return bubble_locations


def plot_channel_responses(
        model, img, ch_idx, dev, ch_mean, ch_std, cont_int_scale, item=None):
    """
    Debugging plot. PLots Image, contour integration input and output

    cont_int_scale = the size reduction between input and the contour integration layer
    """
    f, ax_arr = plt.subplots(1, 3, figsize=(21, 7))

    label_out = process_image(model, dev, ch_mean, ch_std, img)

    img = np.transpose(img.squeeze(), axes=(1, 2, 0))
    ax_arr[0].imshow(img)
    if item:
        ax_arr[0].scatter(
            item.position[1] * cont_int_scale, item.position[0] * cont_int_scale,
            marker='+', color='green', s=120)

    tgt_ch_in_acts = cont_int_in_act[0, ch_idx, :, :]
    ax_arr[1].imshow(tgt_ch_in_acts)
    ax_arr[1].set_title("In.")
    if item:
        ax_arr[1].scatter(
            item.position[1], item.position[0], marker='+', color='red', s=120)
        ax_arr[1].set_title("In. @ {}\n Current {:0.4f},\n Stored {:0.4f}".format(
            item.position, tgt_ch_in_acts[item.position[0], item.position[1]], item.in_act))

    tgt_ch_out_acts = cont_int_out_act[0, ch_idx, :, :]
    ax_arr[2].imshow(tgt_ch_out_acts)
    ax_arr[2].set_title("Out.")
    if item:
        ax_arr[2].scatter(item.position[1], item.position[0], marker='+', color='red', s=120)

        # Plot the contour over the output activation
        ax_arr[2].set_title("Out. @ {}\n Current {:0.4f},\n Stored ={:0.4f}".format(
            item.position,  tgt_ch_out_acts[item.position[0], item.position[1]], item.out_act))

        ax_arr[2].scatter(
            item.ep1[1] // cont_int_scale, item.ep1[0] // cont_int_scale,
            marker='o', color='magenta', s=60)
        ax_arr[2].scatter(
            item.ep2[1] // cont_int_scale, item.ep2[0] // cont_int_scale,
            marker='o', color='magenta', s=60)
        ax_arr[2].scatter(
            item.c1[:, 1] // cont_int_scale, item.c1[:, 0] // cont_int_scale,
            marker='.', color='magenta')

    title = "Prediction {:0.4f}".format(label_out.item())
    if item:
        title = \
            "GT={}, prediction={:0.4f}, Stored  prediction {:0.4f}, Out max act = {:0.4f}," \
            " position {}".format(
                item.gt, label_out.item(), item.prediction, item.out_act, item.position)

    f.suptitle(title)


def find_best_stimuli_for_each_channel(
        model, data_loader, top_n, n_channels, ch_mean, ch_std, n_epochs, cont_int_scale):
    """
    Parse the data loader n_epochs times storing the top n images and other
    details (max activeElement) for each channel of the contour integration layer

    cont_int_scale = the size reduction between input and the contour integration layer

    @ return: A list of TopNTracker objects one for each channel
    """
    func_start_time = datetime.now()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    top_n_per_channel_trackers = [TopNTracker(top_n) for _ in range(n_channels)]
    n_images = len(data_loader)

    for epoch in range(n_epochs):
        for iteration, data_loader_out in enumerate(data_loader, 0):

            print("Epoch {} Iteration {}".format(epoch, iteration))

            if data_loader_out[0].dim() == 4:  # if valid image

                disable_print()
                img, label, sep_c_label, full_label, d, org_img_idx, \
                    c1, c2, start_point, end_point = data_loader_out
                enable_print()

                label_out = process_image(model, dev, ch_mean, ch_std, img)

                # Remove batch dimension
                label = np.squeeze(label)
                # sep_c_label = np.squeeze(sep_c_label)
                # full_label = np.squeeze(full_label)
                # d = np.squeeze(d)
                org_img_idx = np.squeeze(org_img_idx)
                c1 = np.squeeze(c1)
                c2 = np.squeeze(c2)
                start_point = np.squeeze(start_point)
                end_point = np.squeeze(end_point)

                # Only consider connected samples
                if label:
                    for ch_idx in range(n_channels):

                        # Target channel activation
                        curr_tgt_ch_acts = cont_int_out_act[0, ch_idx, :, :]
                        curr_max_act = np.max(curr_tgt_ch_acts)

                        curr_max_act_idx = np.argmax(curr_tgt_ch_acts)  # 1d index
                        curr_max_act_idx = np.unravel_index(
                            curr_max_act_idx, curr_tgt_ch_acts.shape)  # 2d idx
                        curr_max_act_idx = np.array(curr_max_act_idx)

                        # Check for valid sample:
                        # 1. Endpoints should be connected
                        # 2. max_active should be at most one pixel away from the contour
                        min_d_to_contour = np.min(
                            data_loader.dataset.get_distance_point_and_contour(
                                curr_max_act_idx * cont_int_scale, c1))

                        d_ep1 = data_loader.dataset.get_distance_between_two_points(
                            curr_max_act_idx * cont_int_scale, start_point)
                        d_ep2 = data_loader.dataset.get_distance_between_two_points(
                            curr_max_act_idx * cont_int_scale, end_point)

                        if min_d_to_contour < 2 and \
                                d_ep1.item() >= float(data_loader.dataset.end_stop_radius) and \
                                d_ep2.item() >= float(data_loader.dataset.end_stop_radius):

                            # print("Adding img to channel {} top-n list. Distance to contour "
                            #       "{:0.2f}, to ep1 {:0.2f}, ep2 {:0.2f}".format(
                            #         ch_idx, min_d_to_contour, d_ep1, d_ep2))

                            node = MaxActiveElement(
                                in_act=cont_int_in_act[
                                    0, ch_idx, curr_max_act_idx[0], curr_max_act_idx[1]],
                                out_act=curr_max_act,
                                position=curr_max_act_idx,
                                index=org_img_idx,
                                c1=c1,
                                c2=c2,
                                ep1=start_point,  # get rid of batch dim
                                ep2=end_point,
                                prediction=label_out.item(),
                                gt=label
                            )

                            top_n_per_channel_trackers[ch_idx].push(
                                curr_max_act, (n_images * epoch + iteration), node)

    print("Finding Optimal stimuli took {}".format(datetime.now() - func_start_time))

    return top_n_per_channel_trackers


def get_averaged_results(mu_mat, std_mat):
    """
    Average list of averages as if they are from the same RV.
    Average across the channel dimension (axis=0)
    Each entry itself is averaged value. We want to get the average mu and sigma as if they are
    from the same RV

    REF: https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation

    :param mu_mat: [n_channels x n_spacing]
    :param std_mat: [n_channels x n_spacing]

    :return:
    """
    mean_gain = np.mean(mu_mat, axis=0)

    # For Two RVs, X and Y
    # Given mu_x, mu_y, sigma_x, sigma_y
    # sigma (standard deviation) of X + Y = np.sqrt(sigma_x**2 + sigma_y**2)
    # This gives the standard deviation of the sum, of X+Y, to get the average variance
    # if all samples were from same RV, just average the summed variance. Then sqrt it to
    # get avg std
    n = mu_mat.shape[0]

    sum_var = np.sum(std_mat ** 2, axis=0)
    avg_var = sum_var / n
    std_gain = np.sqrt(avg_var)

    return mean_gain, std_gain


def plot_activations(x, in_mean, in_std, out_mean, out_std, title=None):
    """
    Plot Input & Output activations vs RCD
    """
    fig, axis = plt.subplots(figsize=(11, 11))

    axis.plot(x, in_mean, color='r', label='In')
    axis.fill_between(x, in_mean - in_std, in_mean + in_std, alpha=0.2, color='r')

    axis.plot(x, out_mean, color='b', label='Out')
    axis.fill_between(x, out_mean - out_std, out_mean + out_std, alpha=0.2, color='b')

    if title:
        axis.set_title(title)
    axis.set_xlabel('Spacing (relative co-linear distance)')
    axis.set_ylabel("Activation")
    axis.grid()
    axis.legend()

    return fig, axis


def plot_out_all_vs_out_0_gains(x, out_acts, epsilon, title=None):
    """
    plot gain vs RCD
    Gain = Output various / (Output x @ index 0 + epsilon).
    Assumes results for RCD=1 are in the first column
    """
    fig, axis = plt.subplots(figsize=(11, 11))

    # Output response to RCD = 1
    rcd_1_responses = out_acts[:, 0]
    rcd_1_responses = np.expand_dims(rcd_1_responses, axis=1)
    gain = out_acts/(rcd_1_responses + epsilon)

    mean_gain = np.mean(gain, axis=0)
    std_gain = np.std(gain, axis=0)

    axis.plot(x, mean_gain)
    axis.fill_between(x, mean_gain - std_gain, mean_gain + std_gain, alpha=0.2)

    if title:
        axis.set_title(title)
    axis.set_xlabel('Spacing (relative co-linear distance)')
    axis.set_ylabel("Gain [Output various RCD/ (Out RCD=1 + {})]".format(epsilon))
    axis.grid()

    return fig, axis


def plot_out_vs_in_gains(x, in_acts, out_acts, epsilon, title=None):
    """
    Gain = Output Act / (Input Act + epsilon)
    """
    fig, axis = plt.subplots(figsize=(11, 11))
    gain = out_acts / (in_acts + epsilon)

    mean_gain = np.mean(gain, axis=0)
    std_gain = np.std(gain, axis=0)

    axis.plot(x, mean_gain)
    axis.fill_between(x, mean_gain - std_gain, mean_gain + std_gain, alpha=0.2)

    if title:
        axis.set_title(title)
    axis.set_xlabel('Spacing (relative co-linear distance)')
    axis.set_ylabel("Gain [Output/(Input + {})]".format(epsilon))
    axis.grid()

    return fig, axis


def plot_predictions(x, preds_mat, title=None):
    """
    PLot predictions vs RCD
    """
    fig, axis = plt.subplots(figsize=(11, 11))

    mean_preds = preds_mat.mean(axis=0)
    std_preds = preds_mat.std(axis=0)

    axis.plot(x, mean_preds)
    axis.fill_between(x, mean_preds - std_preds, mean_preds + std_preds, alpha=0)

    if title:
        axis.set_title(title)
    axis.set_xlabel('Spacing (Relative co-linear distance)')
    axis.set_ylabel('Avg Prediction')
    axis.grid()

    return fig, axis


def plot_histogram_of_linear_fit_gradients(x, mean_in_acts, mean_out_acts):
    """
    For each channel plot the histogram of gradients for the linear fit for each channel
    # different way of showing population results
     mean_in_acts = [n_channels x n_RCD]
    """
    n_channels = len(mean_in_acts)

    in_acts_gradients = []
    out_acts_gradients = []

    for ch_idx in range(n_channels):
        in_acts = mean_in_acts[ch_idx, ]
        out_acts = mean_out_acts[ch_idx, ]

        m_in, b_in = np.polyfit(x, in_acts, deg=1)
        m_out, b_out = np.polyfit(x, out_acts, deg=1)

        in_acts_gradients.append(m_in)
        out_acts_gradients.append(m_out)

    f, ax_arr = plt.subplots(1, 2, figsize=(11, 11))

    ax_arr[0].hist(m_in)
    ax_arr[0].set_xlabel("Linear fit Gradient")
    ax_arr[0].set_title("Gradients of linear fits to Input act vs RCD")

    ax_arr[1].hist(m_out)
    ax_arr[1].set_xlabel("Linear fit Gradient")
    ax_arr[1].set_title("Gradients of linear fits to Output Act vs RCD")

    return f, ax_arr


def plot_tiled_activations(x, mean_in_acts, mean_out_acts):
    """
    Plot mean input/output activation per channel (dim=0) individually in a tiled image.
    mean_in_acts = [n_channels x n_RCD]
    """
    n_channels = len(mean_in_acts)
    tile_single_dim = np.int(np.ceil(np.sqrt(n_channels)))

    f, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(11, 11))

    for ch_idx in range(n_channels):
        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        ax_arr[r_idx, c_idx].plot(x, mean_in_acts[ch_idx, ], label='in', color='r')
        ax_arr[r_idx, c_idx].plot(x, mean_out_acts[ch_idx, ], label='out', color='b')
        # ax_arr[r_idx, c_idx].axis('off')  # Turn off all labels

    f.suptitle("Individual Neuron Activations [Red=Input, Blue=Output]")

    return f, ax_arr


def plot_tiled_out_all_vs_out_0_gains(x, mean_out_acts, epsilon):
    """
    Plot gain individually for each channel in a tiled image.
    Gain = Output various / (Output x @ index 0 + epsilon).
    Assumes results for RCD=1 are in the first column

    mean_out_acts = [n_channels x n_RCD]
    """
    n_channels = len(mean_out_acts)
    tile_single_dim = np.int(np.ceil(np.sqrt(n_channels)))

    f, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(11, 11))

    for ch_idx in range(n_channels):
        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        rcd_1_responses = mean_out_acts[ch_idx, 0]

        ax_arr[r_idx, c_idx].plot(
            x, mean_out_acts[ch_idx] / (epsilon + rcd_1_responses), label='gain')
        # ax_arr[r_idx, c_idx].axis('off')  # Turn off all labels

    f.suptitle("Individual Neuron Gain [Output various RCD/ (Out RCD=1 + {})]".format(epsilon))

    return f, ax_arr


def plot_tiled_out_vs_in_gains(x, mean_in_acts, mean_out_acts, epsilon):
    """
    Plot gain individually for each channel in a tiled image.
    """
    n_channels = len(mean_in_acts)
    tile_single_dim = np.int(np.ceil(np.sqrt(n_channels)))

    f, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(11, 11))

    for ch_idx in range(n_channels):
        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        ax_arr[r_idx, c_idx].plot(
            x, mean_out_acts[ch_idx] / (epsilon + mean_in_acts[ch_idx, ]), label='gain')
        # ax_arr[r_idx, c_idx].axis('off')  # Turn off all labels

    f.suptitle("Individual Neuron Gain [Output / (Input + {}]".format(epsilon))

    return f, ax_arr


def plot_tiled_predictions(x, mean_preds, std_preds):
    """
    plot predictions vs RCD for each channel individually in a  tiled image

    mean_preds = [n_channels x n_rcd]

    """
    n_channels = len(mean_preds)
    tile_single_dim = np.int(np.ceil(np.sqrt(n_channels)))

    f, ax_arr = plt.subplots(tile_single_dim, tile_single_dim, figsize=(11, 11))

    for ch_idx in range(n_channels):
        r_idx = ch_idx // tile_single_dim
        c_idx = ch_idx - r_idx * tile_single_dim

        ax_arr[r_idx, c_idx].plot(x, mean_preds[ch_idx], label='gain')
        ax_arr[r_idx, c_idx].fill_between(
            x, mean_preds[ch_idx] - std_preds[ch_idx], mean_preds[ch_idx] + std_preds[ch_idx],
            alpha=0.2)
        # ax_arr[r_idx, c_idx].axis('off')  # Turn off all labels

    f.suptitle("Individual Predictions")

    return f, ax_arr


def plot_population_average_results(
        x, mean_in_acts, std_in_acts, mean_out_acts, std_out_acts, epsilon_oi_gain,
        epsilon_oo_gain):
    """
    PLot (1) Average activation across channels
         (2) Average gain (input vs output) gain
         (3) Average gain (output all vs out 0) gain

    epsilon_io_gain = epsilon for input output gain
    epsilon_oo_gain = epsilon for output output gain

    mean_in_acts = [n_channels x n_rcd]  where each row is the avg response across many images
    """

    pop_mean_in_act, pop_std_in_act = get_averaged_results(mean_in_acts, std_in_acts)
    pop_mean_out_act, pop_std_out_act = get_averaged_results(mean_out_acts, std_out_acts)

    f, ax_arr = plt.subplots(1, 3, figsize=(11, 11))

    # Plot average activations
    ax_arr[0].plot(x, pop_mean_in_act, label='In')
    ax_arr[0].fill_between(
        x, pop_mean_in_act - pop_std_in_act, pop_mean_in_act + pop_std_in_act, alpha=0.2)
    ax_arr[0].plot(x, pop_mean_out_act, label='Out')
    ax_arr[0].fill_between(
        x, pop_mean_out_act - pop_std_out_act, pop_mean_out_act + pop_std_out_act, alpha=0.2)
    ax_arr[0].set_xlabel('Spacing (relative co-linear distance)')
    ax_arr[0].set_ylabel("Activations")
    ax_arr[0].legend()
    ax_arr[0].grid()

    # Plot output vs output Gains
    oo_gains = mean_out_acts / (pop_mean_out_act[0] + epsilon_oo_gain)
    # TODO: Check if the STD of population gain is correct
    pop_mean_gains = np.mean(oo_gains, axis=0)
    pop_std_gains = np.std(oo_gains, axis=0)
    ax_arr[1].plot(x, pop_mean_gains)
    ax_arr[1].fill_between(
        x, pop_mean_gains - pop_std_gains, pop_mean_gains + pop_std_gains, alpha=0.2)
    ax_arr[1].set_xlabel('Spacing (relative co-linear distance)')
    ax_arr[1].set_ylabel("Gain (Out RCD=all/(Out RCD=1 + {})".format(epsilon_oo_gain))
    ax_arr[1].grid()

    # Plot output vs input gain
    io_gains = mean_out_acts / (mean_in_acts + epsilon_oi_gain)
    pop_mean_gains = np.mean(io_gains, axis=0)
    pop_std_gains = np.std(io_gains, axis=0)
    ax_arr[2].plot(x, pop_mean_gains)
    ax_arr[2].fill_between(
        x, pop_mean_gains - pop_std_gains, pop_mean_gains + pop_std_gains, alpha=0.2)
    ax_arr[2].set_xlabel('Spacing (relative co-linear distance)')
    ax_arr[2].set_ylabel("Gain (Out/(In + {})".format(epsilon_oi_gain))
    ax_arr[2].grid()

    return f, ax_arr


def plot_population_average_predictions(x, mean_pred_per_ch, std_pred_per_ch):
    """
    mean_pred_per_ch = [n_channels, n_rcd]
    """
    f, ax = plt.subplots()

    pop_mean_pred, pop_std_pred = get_averaged_results(mean_pred_per_ch, std_pred_per_ch)

    ax.plot(x, pop_mean_pred)
    ax.fill_between(
        x, pop_mean_pred - pop_std_pred, pop_mean_pred + pop_std_pred, alpha=0.2)
    ax.set_xlabel('Spacing (relative co-linear distance)')
    ax.set_ylabel("predictions")
    ax.grid()

    return f, ax


def debug_plot_contour_and_bubble_locations(
        img, item, bubble_loc_arr, closest_contour_point_idx, cont_int_scale):
    """
    On top the input image plot contour, max active neuron location,
    contour and location of bubbles
    """

    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(img, axes=(1, 2, 0)))

    upscaled_position = np.array(item.position) * cont_int_scale
    plt.scatter(
        upscaled_position[1], upscaled_position[0],
        marker='d', color='yellow', s=120, label='max active neuron position')

    closest_contour_point = item.c1[closest_contour_point_idx]
    plt.scatter(
        closest_contour_point[1], closest_contour_point[0],
        marker='+', color='yellow', s=120, label='closest_contour_point')

    plt.scatter(
        item.c1[:, 1], item.c1[:, 0],
        marker='.', color='magenta', label='contour')

    plt.scatter(
        bubble_loc_arr[:, 1], bubble_loc_arr[:, 0],
        marker='o', color='yellow', s=120, label='bubble centers')

    plt.legend()


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main(model, base_results_dir, data_set_params, cont_int_scale, top_n=50, n_channels=64):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    frag_tile_size = np.array([7, 7])
    bubble_tile_sizes = np.array([[7, 7], [9, 9], [11, 11], [13, 13], [15, 15], [17, 17]])

    # For gain calculation to prevent divide by zero,
    epsilon_gain_oi = 1e-2  # for gain = Output / (input + epsilon_gain_oi)
    epsilon_gain_oo = 1e-4  # for gain = Outputs / (output RCD=1 + epsilon_gain_oo)

    required_data_set_params = [
        'biped_dataset_dir', 'biped_dataset_type', 'n_biped_imgs', 'n_epochs']
    for param in required_data_set_params:
        if param not in data_set_params:
            raise Exception("Required Dataset Param {} not found".format(param))

    # Immutable
    # ---------
    np.set_printoptions(precision=3)

    # Relative co-linear distance =  spacing / fragment length
    rcd = bubble_tile_sizes[:, 0] / np.float(frag_tile_size[0])
    total_n_imgs = data_set_params['n_biped_imgs'] * data_set_params['n_epochs']

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # Results folder: structure
    #   experiment_gain_vs_frag_size_natural_images
    #       individual_channels
    #           io_gains
    #           oo_gains
    #           activations
    #           predictions

    results_dir = os.path.join(
        base_results_dir, 'experiment_gain_vs_frag_size_natural_images_test')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    idv_channels_results_dir = os.path.join(results_dir, 'individual_channels')
    if not os.path.exists(idv_channels_results_dir):
        os.makedirs(idv_channels_results_dir)

    idv_acts_dir = os.path.join(idv_channels_results_dir, 'activations')
    if not os.path.exists(idv_acts_dir):
        os.makedirs(idv_acts_dir)

    idv_oi_gains_dir = os.path.join(idv_channels_results_dir, 'oi_gains')
    if not os.path.exists(idv_oi_gains_dir):
        os.makedirs(idv_oi_gains_dir)

    idv_oo_gains_dir = os.path.join(idv_channels_results_dir, 'oo_gains')
    if not os.path.exists(idv_oo_gains_dir):
        os.makedirs(idv_oo_gains_dir)

    idv_preds_dir = os.path.join(idv_channels_results_dir, 'predictions')
    if not os.path.exists(idv_preds_dir):
        os.makedirs(idv_preds_dir)

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    data_set = OnlineNaturalImagesPathfinder(
        data_dir=data_set_params['biped_dataset_dir'],
        dataset_type=data_set_params['biped_dataset_type'],
        transform=None,  # Normalize each image individually, as part of process_image
        subset_size=data_set_params['n_biped_imgs'],
        resize_size=(256, 256),
        p_connect=1,  # Only interested in connected samples
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=1,  # Has to be 1, returned contours are of different sizes
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Find Optimal stimuli
    # -----------------------------------------------------------------------------------
    print("Finding Optimal Stimuli for each Channel. Num Images {} ...".format(total_n_imgs))
    top_n_per_channel_trackers = find_best_stimuli_for_each_channel(
        model, data_loader, top_n, n_channels, ch_mean, ch_std,
        data_set_params['n_epochs'], cont_int_scale)

    # -----------------------------------------------------------------------------------
    # Effect of Fragment spacing
    # -----------------------------------------------------------------------------------
    print("Finding responses to fragmented contours ...")

    summary_file = os.path.join(results_dir, 'summary.txt')
    f_handle = open(summary_file, 'w+')

    f_handle.write("Settings {}\n".format('-' * 80))
    f_handle.write("Fragment Length {}\n".format(frag_tile_size[0]))
    f_handle.write("Bubble Lengths  {}\n".format([x[0] for x in bubble_tile_sizes]))
    f_handle.write("Results {}\n".format('-' * 80))

    # Variables to track across all channels
    mean_out_acts = np.zeros((n_channels, len(rcd)))
    std_out_acts = np.zeros_like(mean_out_acts)
    mean_in_acts = np.zeros_like(mean_out_acts)
    std_in_acts = np.zeros_like(mean_out_acts)
    mean_preds = np.zeros_like(mean_out_acts)
    std_preds = np.zeros_like(mean_out_acts)
    n_images_list = []  # Number of images averaged for each channel

    for ch_idx in range(n_channels):
        n_images = len(top_n_per_channel_trackers[ch_idx])
        n_images_list.append(n_images)
        if n_images == 0:
            print("No stored images for channel {}".format(ch_idx))
            continue

        print("Finding Contour Gain for Channel {}. Number of Stored Images {}".format(
            ch_idx, n_images))

        # Get the store Images
        max_active_nodes, _ = top_n_per_channel_trackers[ch_idx].get_stored_values()

        # Variables to track per image
        tgt_n_in_act_mat = np.zeros((n_images, len(bubble_tile_sizes)))
        tgt_n_out_act_mat = np.zeros_like(tgt_n_in_act_mat)
        tgt_n_pred_mat = np.ones_like(tgt_n_in_act_mat) * -1  # invalid value
        # (in, out) activations for image with continuous contour
        tgt_n_cont_c_acts = np.zeros((n_images, 2))

        for item_idx, item in enumerate(max_active_nodes):
            # Find the closest point on contour to max active point
            d_to_contour = \
                data_set.get_distance_point_and_contour(
                    np.array(item.position) * cont_int_scale, item.c1)

            closest_contour_point_idx = np.argmin(d_to_contour)

            tgt_n_cont_c_acts[item_idx, ] = np.array([item.in_act, item.out_act])

            for bubble_tile_idx, bubble_tile_size in enumerate(bubble_tile_sizes):
                # Create punctured image
                # ----------------------
                # Bubbles are placed only on the contour.
                bubble_center_locations = \
                    get_bubbles_locations(
                        item.c1.numpy(),
                        closest_contour_point_idx,
                        frag_tile_size[0],
                        bubble_tile_size[0], data_set)

                # Puncture bubbles locations are specified by the right corner of
                # the bubble tile size
                puncture = utils.PunctureImage(
                    n_bubbles=1, fwhm=bubble_tile_size[0], tile_size=bubble_tile_size)
                bubble_insert_locations = bubble_center_locations - bubble_tile_size // 2

                punctured_img = puncture(
                    data_set.get_img_by_index(item.index),
                    start_loc_arr=bubble_insert_locations)

                data_set.add_end_stop(punctured_img, item.ep1)
                data_set.add_end_stop(punctured_img, item.ep2)

                # Process the image
                # -----------------
                label_out = process_image(model, dev, ch_mean, ch_std, punctured_img)
                r = item.position[0]
                c = item.position[1]

                # Store inputs and outputs, no gain calculations
                tgt_n_in_act_mat[item_idx, bubble_tile_idx] = cont_int_in_act[0, ch_idx, r, c]
                tgt_n_out_act_mat[item_idx, bubble_tile_idx] = cont_int_out_act[0, ch_idx, r, c]
                tgt_n_pred_mat[item_idx, bubble_tile_idx] = label_out

                # # Debug: PLot contour and bubble locations
                # debug_plot_contour_and_bubble_locations(
                #     punctured_img, item, bubble_center_locations, closest_contour_point_idx,
                #     cont_int_scale)
                # import pdb
                # pdb.set_trace()

                # # Debug : Display the Image
                # # -------------------------
                # if bubble_tile_idx == 0:
                #     org_img = data_set.get_img_by_index(item.index, item.ep1, item.ep2)
                #     plot_channel_responses(
                #         model, org_img, ch_idx, dev, ch_mean, ch_std, cont_int_scale, item)
                #     plt.gcf().suptitle('Original Image\n' + plt.gcf()._suptitle.get_text())
                #
                # plot_channel_responses(
                #     model, punctured_img, ch_idx, dev, ch_mean, ch_std, cont_int_scale, item)
                # plt.gcf().suptitle(
                #     'Punctured Image {}\n'.format(bubble_tile_size[0]) +
                #     plt.gcf()._suptitle.get_text())

            # -----------------------------
            # import pdb
            # pdb.set_trace()
            # plt.close('all')

        # Per channel processing --------------------------------------------------------
        # Save the results
        mean_in_acts[ch_idx, ] = tgt_n_in_act_mat.mean(axis=0)
        std_in_acts[ch_idx, ] = tgt_n_in_act_mat.std(axis=0)
        mean_out_acts[ch_idx, ] = tgt_n_out_act_mat.mean(axis=0)
        std_out_acts[ch_idx, ] = tgt_n_out_act_mat.std(axis=0)
        mean_preds[ch_idx, ] = tgt_n_pred_mat.mean(axis=0)
        std_preds[ch_idx, ] = tgt_n_pred_mat.std(axis=0)

        # Per Channel Plots
        # Activations
        f, ax = plot_activations(
            rcd, mean_in_acts[ch_idx, ], std_in_acts[ch_idx, ],
            mean_out_acts[ch_idx, ], std_in_acts[ch_idx, ],
            title=("Channel {}. Number of images {}".format(ch_idx, n_images))
        )

        mean_tgt_n_cont_c_acts = np.mean(tgt_n_cont_c_acts, axis=0)
        std_tgt_n_cont_c_acts = np.std(tgt_n_cont_c_acts, axis=0)
        ax.errorbar(
            1, mean_tgt_n_cont_c_acts[0], std_tgt_n_cont_c_acts[0],
            label='continuous contour In', c='r', capsize=10, capthick=3, markersize=10)
        ax.errorbar(
            1, mean_tgt_n_cont_c_acts[1], std_tgt_n_cont_c_acts[1],
            label='continuous Contour Out', c='b', capsize=10, capthick=3, markersize=10)
        ax.legend()
        f.savefig(os.path.join(idv_acts_dir, 'activations_channel_{}.png'.format(ch_idx)))

        # Output Input gains
        f, ax = plot_out_vs_in_gains(
            rcd, tgt_n_in_act_mat, tgt_n_out_act_mat, epsilon_gain_oi,
            title="Gain = Output / (Input+ {}).\n Channel {}. Number of images {}".format(
                epsilon_gain_oi, ch_idx, n_images))
        f.savefig(os.path.join(idv_oi_gains_dir, 'oi_gains_channel_{}.png'.format(ch_idx)))

        f, ax = plot_out_all_vs_out_0_gains(
            rcd, tgt_n_out_act_mat, epsilon_gain_oo,
            title="Gain = Output / (Output RCD=1 + {}).\n Channel {}. Number of images {}".format(
                epsilon_gain_oo, ch_idx, n_images))
        f.savefig(os.path.join(idv_oo_gains_dir, 'oo_gains_channel_{}.png'.format(ch_idx)))

        # Predictions
        f, ax = plot_predictions(
            rcd, tgt_n_pred_mat, title="Channel {}. Number of images {}".format(ch_idx, n_images))
        f.savefig(os.path.join(idv_preds_dir, 'preds_channel_{}.png'.format(ch_idx)))

        plt.close('all')

    # Population Results -------------------------
    print("Mean In Activations: \n" + 'np.' + repr(mean_in_acts), file=f_handle)
    print("Std In Activations: \n" + 'np.' + repr(mean_in_acts), file=f_handle)
    print("Mean Out Activations: \n" + 'np.' + repr(mean_out_acts), file=f_handle)
    print("Std Out Activations: \n" + 'np.' + repr(mean_out_acts), file=f_handle)

    # Overall Plots
    # Tiled Individual channels Activations
    f, ax_arr = plot_tiled_activations(rcd, mean_in_acts, mean_out_acts)
    f.savefig(os.path.join(results_dir, 'individual_channel_activations.jpg'), format='jpg')
    # Tiled Output Input Gains
    f, ax_arr = plot_tiled_out_vs_in_gains(rcd, mean_in_acts, mean_out_acts, epsilon_gain_oi)
    f.savefig(os.path.join(results_dir, 'individual_channel_oi_gains.jpg'), format='jpg')
    # Tiled Output Output Gains
    f, ax_arr = plot_tiled_out_all_vs_out_0_gains(rcd, mean_out_acts, epsilon_gain_oo)
    f.savefig(os.path.join(results_dir, 'individual_channel_oo_gains.jpg'), format='jpg')
    # Tiled predictions
    f, ax_arr = plot_tiled_predictions(rcd, mean_preds, std_preds)
    f.savefig(os.path.join(results_dir, 'individual_channel_predictions.jpg'), format='jpg')

    # Population Average Results
    # Population Avg activations and gains
    f, ax_arr = plot_population_average_results(
        rcd, mean_in_acts, std_in_acts, mean_out_acts, std_out_acts,
        epsilon_gain_oi, epsilon_gain_oo)
    f.savefig(os.path.join(results_dir, 'population_results.jpg'), format='jpg')
    # Population average predictions
    f, ax_arr = plot_population_average_predictions(rcd, mean_preds, std_preds)
    f.savefig(os.path.join(results_dir, 'population_predictions.jpg'), format='jpg')
    # Distribution of gradients of gain vs rcd curves
    f, ax_arr = plot_histogram_of_linear_fit_gradients(rcd, mean_in_acts, mean_out_acts)
    f.savefig(os.path.join(results_dir, 'histogram_of_gradient_fits.jpg'), format='jpg')

    plt.close('all')
# ---------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Initialization
    # --------------
    random_seed = 7
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    plt.ion()
    start_time = datetime.now()

    # Dataset Parameters
    dataset_parameters = {
        'biped_dataset_dir': './data/BIPED/edges',
        'biped_dataset_type': 'train',
        'n_biped_imgs': 20000,
        'n_epochs': 1  # Total images = n_epochs * n_biped_images
    }

    # # Model
    # # ------
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    #
    # net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    # saved_model = \
    #     './results/pathfinder/' \
    #     'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20200807_214306_base/' \
    #     'best_accuracy.pth'
    # scale_down_input_to_contour_integration_layer = 4

    # Control Model
    # -----
    cont_int_layer = new_control_models.ControlMatchParametersLayer(
        lateral_e_size=15, lateral_i_size=15)
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    saved_model = \
        './results/pathfinder/' \
        'BinaryClassifierResnet50_ControlMatchParametersLayer_20200807_214527_base/' \
        'best_accuracy.pth'
    scale_down_input_to_contour_integration_layer = 4

    results_store_dir = os.path.dirname(saved_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))
    main(
        net,
        results_store_dir,
        data_set_params=dataset_parameters,
        cont_int_scale=scale_down_input_to_contour_integration_layer
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))

    import pdb
    pdb.set_trace()
