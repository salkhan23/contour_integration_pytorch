# ---------------------------------------------------------------------------------------
# Contour Gain/Spacing variant for natural images
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import heapq

import torch
from torchvision import transforms

import models.new_piech_models as new_piech_models
from generate_pathfinder_dataset import OnlineNaturalImagesPathfinder
from torch.utils.data import DataLoader
import utils


edge_extract_act = []
cont_int_in_act = []
cont_int_out_act = []


def edge_extract_cb(self, layer_in, layer_out):
    """ Attach at Edge Extract layer
        Callback to Retrieve the activations output of edge Extract layer
    """
    global edge_extract_act
    edge_extract_act = layer_out.cpu().detach().numpy()


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer
        Callback to Retrieve the activations input & output of the contour Integration layer
    """
    global cont_int_in_act
    global cont_int_out_act

    cont_int_in_act = layer_in[0].cpu().detach().numpy()
    cont_int_out_act = layer_out.cpu().detach().numpy()


def process_image(model, devise_to_use, ch_mus, ch_sigmas, in_img):
    """
    Pass image through model and get iou score of the prediction if in_img_label is not None

    :param model:
    :param devise_to_use:
    :param in_img:
    :param ch_mus:
    :param ch_sigmas:
    :return:
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
    """ An element in the Top-n max Active Dictionary """
    def __init__(self, activation, position, index, c1, c2, ep1, ep2, prediction, gt):
        self.activation = activation
        self.position = position
        self.index = index  # index of the original image
        self.c1 = c1
        self.c2 = c2
        self.ep1 = ep1
        self.ep2 = ep2
        self.prediction = prediction
        self.gt = gt

    def print(self):
        print("Max act {:0.4f}. position {}, original image idx {}. Prediction {:0.4f}".format(
            self.activation, self.position, self.index, self.prediction))


class TopNTracker(object):
    """ Use a priority Queue, to keep track of 10 n values"""

    def __init__(self, depth=5):
        """
        """
        self._heap = []
        self.depth = depth

    def push(self, value, count, item):
        """
        The count variable is added to make items unique in case max activations are equal.
        In which case, heappq will try to compare the next item in the tuple (MaxActiveElement)
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
            # reverse the order, max first.
            zipped_object = zip(lst_values, lst_items)
            zipped_object = sorted(zipped_object, reverse=True)

            unzipped_object = zip(*zipped_object)
            lst_values, lst_items = list(unzipped_object)

        return lst_items, lst_values


def get_closest_distance_and_index(desired_d, d_arr):
    """
    """
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
    to find insert location of bubbles that result in a contour with fragmes of size fragment
    length and separated by bubble lengths

    The first visible fragment is centered  at the starting point
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


def plot_channel_responses(model, img, ch_idx, dev, ch_mean, ch_std, item=None):
    """" Debugging plot"""
    f, ax_arr = plt.subplots(1, 3, figsize=(21, 7))

    label_out = process_image(model, dev, ch_mean, ch_std, img)

    img = np.transpose(img.squeeze(), axes=(1, 2, 0))
    ax_arr[0].imshow(img)

    tgt_ch_in_acts = cont_int_in_act[0, ch_idx, :, :]
    ax_arr[1].imshow(tgt_ch_in_acts)
    ax_arr[1].set_title("In.")
    if item:
        ax_arr[2].scatter(item.position[1], item.position[0], marker='o', color='magenta', s=120)
        ax_arr[2].set_title("In.\n @ {} ={:0.4f}[stored]".format(
            item.position, tgt_ch_in_acts[item.position[0], item.position[1]]))

    tgt_ch_out_acts = cont_int_out_act[0, ch_idx, :, :]
    max_act_idx = np.argmax(tgt_ch_out_acts)  # 1d index
    max_act_idx = np.unravel_index(max_act_idx, tgt_ch_out_acts.shape)  # 2d i
    ax_arr[2].imshow(tgt_ch_out_acts)
    ax_arr[2].set_title("Out\n. Max = {:.4f} @ {}".format(np.max(tgt_ch_out_acts), max_act_idx))
    ax_arr[2].scatter(max_act_idx[1], max_act_idx[0], marker='+', color='red', s=120)

    if item:
        ax_arr[2].scatter(item.position[1], item.position[0], marker='o', color='magenta', s=120)
        ax_arr[2].set_title("Out.\n @ {} ={:0.4f} [Current max] \n @ {} ={:0.4f}[stored]".format(
            max_act_idx, np.max(tgt_ch_out_acts), item.position,
            tgt_ch_out_acts[item.position[0], item.position[1]]))
    #     ax_arr[2].scatter(item.ep1[1] // 4, item.ep1[0] // 4, marker='o', color='magenta', s=60)
    #     ax_arr[2].scatter(item.ep2[1] // 4, item.ep2[0] // 4, marker='o', color='magenta', s=60)
    #     ax_arr[2].scatter(item.c1[:, 1] // 4, item.c1[:, 0] // 4, marker='.', color='magenta')

    title = "Prediction {:0.4f}".format(label_out.item())
    if item:
        title = \
            "GT={}, prediction={:0.4f}, Stored  prediction {:0.4f}, Out max act = {:0.4f}," \
            " position {}".format(
                item.gt, label_out.item(), item.prediction, item.activation, item.position)

    f.suptitle(title)


def main(model, base_results_dir):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    n_channels = 64
    top_n = 50

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    biped_dataset_dir = './data/BIPED/edges'
    biped_dataset_type = 'train'
    n_biped_imgs = 20000

    data_set = OnlineNaturalImagesPathfinder(
        data_dir=biped_dataset_dir,
        dataset_type=biped_dataset_type,
        transform=None,
        subset_size=n_biped_imgs,
        resize_size=(256, 256),
        p_connect=1,
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=1,  # Has to be 1, returned contours are of different sizes
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    # Find top n max responsive images for each channel
    # --------------------------------------------------
    print("Finding Optimal Stimuli for each channel ...")

    top_n_per_channel_trackers = [TopNTracker(top_n) for _ in range(n_channels)]

    for iteration, data_loader_out in enumerate(data_loader, 1):

        if data_loader_out[0].dim() == 4:  # if valid image
            img, label, sep_c_label, full_label, d, org_img_idx, \
                c1, c2, start_point, end_point = data_loader_out

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

            if label:  # only consider connected samples

                for ch_idx in range(n_channels):

                    # Target channel activation
                    curr_tgt_ch_acts = cont_int_out_act[0, ch_idx, :, :]
                    curr_max_act = np.max(curr_tgt_ch_acts)

                    curr_max_act_idx = np.argmax(curr_tgt_ch_acts)  # 1d index
                    curr_max_act_idx = np.unravel_index(
                        curr_max_act_idx, curr_tgt_ch_acts.shape)  # 2d idx

                    # Check for valid sample:
                    # 1. Endpoints should be connected
                    # 2. max_active should be at most one pixel away from the contour
                    min_d_to_contour = np.min(
                        data_set.get_distance_point_and_contour(curr_max_act_idx, c1 // 4))

                    if min_d_to_contour < 1.5:
                        node = MaxActiveElement(
                            activation=curr_max_act,
                            position=curr_max_act_idx,
                            index=org_img_idx,
                            c1=c1,
                            c2=c2,
                            ep1=start_point,  # get rid of batch dim
                            ep2=end_point,
                            prediction=label_out.item(),
                            gt=label
                        )

                        top_n_per_channel_trackers[ch_idx].push(curr_max_act, iteration, node)

    # -------------------------------------------------------------------------------
    # Effect of fragment spacing
    # -------------------------------------------------------------------------------
    frag_tile_size = np.array([7, 7])
    bubble_tile_sizes = np.array([[7, 7], [9, 9], [11, 11], [13, 13], [15, 15]])

    results_dir = os.path.join(base_results_dir, 'experiment_gain_vs_frag_size_natural_images')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    idv_channels_results_dir = os.path.join(results_dir, 'individual_channels')
    if not os.path.exists(idv_channels_results_dir):
        os.makedirs(idv_channels_results_dir)

    for ch_idx in range(n_channels):
        print("Finding Contour Gain for Channel {}. Number of Stored Images {}. ...".format(
            ch_idx, len(top_n_per_channel_trackers[ch_idx])))

        ch_results_dir = os.path.join(idv_channels_results_dir, 'channel_{}'.format(ch_idx))
        if not os.path.exists(ch_results_dir):
            os.makedirs(ch_results_dir)

        ch_summary_file = os.path.join(ch_results_dir, 'summary.txt')
        f_handle = open(ch_summary_file, 'w+')
        # f_handle.write("bubble_sizes {}\n".format(bubble_tile_sizes))

        # -------------------------------------------------------------------------------
        max_active_nodes, _ = top_n_per_channel_trackers[ch_idx].get_stored_values()

        # Results to Track
        tgt_n_in_act_mat = np.zeros((len(max_active_nodes), len(bubble_tile_sizes)))
        tgt_n_out_act_mat = np.zeros_like(tgt_n_in_act_mat)
        tgt_n_labels_mat = np.ones_like(tgt_n_in_act_mat) * -1000

        # Process each image
        for item_idx, item in enumerate(max_active_nodes):

            # Find the closest point on contour to max active point
            d_to_contour = data_set.get_distance_point_and_contour(item.position, item.c1 // 4)
            closest_contour_point_idx = np.argmin(d_to_contour)

            for bubble_tile_idx, bubble_tile_size in enumerate(bubble_tile_sizes):
                # Create punctured image
                # -----------------------------------------------------------------------
                bubble_insert_locations = get_bubbles_locations(
                    item.c1.numpy(),
                    closest_contour_point_idx,
                    frag_tile_size[0],
                    bubble_tile_size[0],
                    data_set,)

                # Bubbles are inserted using the position of the top left corner of the tile
                # Note: - bubble_tile_size = full tile_size //2
                bubble_insert_locations = bubble_insert_locations - bubble_tile_size

                puncture = utils.PunctureImage(
                    n_bubbles=1, fwhm=bubble_tile_size[0], tile_size=bubble_tile_size * 2)

                new_img = data_set.get_img_by_index(item.index)
                punctured_img = puncture(new_img, start_loc_arr=bubble_insert_locations)

                data_set.add_end_stop(punctured_img, item.ep1)
                data_set.add_end_stop(punctured_img, item.ep2)

                # # Debug : Display the Image
                # org_img = data_set.get_img_by_index(item.index, item.ep1, item.ep2)
                # plot_channel_responses(model, org_img, ch_idx, dev, ch_mean, ch_std, item)
                # plt.gcf().suptitle('Original Image\n' + plt.gcf()._suptitle.get_text())
                #
                # plot_channel_responses(model, punctured_img, ch_idx, dev, ch_mean, ch_std, item)
                # plt.gcf().suptitle('Punctured Image\n' + plt.gcf()._suptitle.get_text())

                # Process the punctured image
                # -----------------------------------------------------------------------
                label_out = process_image(model, dev, ch_mean, ch_std, punctured_img)
                r = item.position[0]
                c = item.position[1]

                tgt_n_in_act_mat[item_idx, bubble_tile_idx] = cont_int_in_act[0, ch_idx, r, c]
                tgt_n_out_act_mat[item_idx, bubble_tile_idx] = cont_int_out_act[0, ch_idx, r, c]
                tgt_n_labels_mat[item_idx, bubble_tile_idx] = label_out

        # Saves the results
        np.set_printoptions(precision=3)
        f_handle.write("Inputs: \n")
        print(tgt_n_in_act_mat, file=f_handle)
        f_handle.write('\n')

        f_handle.write("Outputs: \n")
        print(tgt_n_out_act_mat, file=f_handle)
        f_handle.write('\n')
        f_handle.close()

        # Plot Input and outputs
        rcd = bubble_tile_sizes[:, 0] / np.float(frag_tile_size[0])  # relative co-linear distance
        f, ax = plt.subplots()
        ax.errorbar(
            rcd,
            tgt_n_in_act_mat.mean(axis=0),
            tgt_n_in_act_mat.std(axis=0),
            label='In act')
        ax.errorbar(
            rcd,
            tgt_n_out_act_mat.mean(axis=0),
            tgt_n_out_act_mat.std(axis=0),
            label='out act')

        ax.set_title("Activations. Channel {}. Number of images {}".format(
            ch_idx, len(max_active_nodes)))
        ax.grid()
        ax.legend()
        f.savefig(os.path.join(ch_results_dir, 'activations_channel_{}.png'.format(ch_idx)))

        # Plot Main Gains
        f, ax = plt.subplots()
        gain = tgt_n_out_act_mat / (tgt_n_in_act_mat + 1e-4)
        ax.errorbar(
            rcd,
            gain.mean(axis=0),
            gain.std(axis=0))
        ax.set_title("Contour Integration Gain Channel {}. Number of images {}".format(
            ch_idx, len(max_active_nodes)))
        ax.grid()
        ax.legend()
        f.savefig(os.path.join(ch_results_dir, 'gain_channel_{}.png'.format(ch_idx)))

        import pdb
        pdb.set_trace()
        plt.close('all')


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 7

    # Immutable
    # ---------
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    plt.ion()
    start_time = datetime.now()

    # Model
    # ------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.JointPathfinderContourResnet50(cont_int_layer)
    saved_model = \
        'results/joint_training/' \
        'JointPathfinderContourResnet50_CurrentSubtractInhibitLayer_20200719_104417_base/' \
        'last_epoch.pth'

    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    # saved_model = \
    #     './results/pathfinder/' \
    #     'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20200716_173915_with_maxpooling/' \
    #     'best_accuracy.pth'

    results_store_dir = os.path.dirname(saved_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))
    main(net, results_store_dir)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb

    pdb.set_trace()
