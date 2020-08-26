# ---------------------------------------------------------------------------------------
# Contour Gain vs contour length variant for natural images.
# For selected contours, placement of end points is varied to study the impact of contour
# length. Occlusion bubbles are used to block out other parts of the contour
#
# Similar to gain_vs_spacing_natural_images.py but uses a different definition for gain.
# Contour Integration Gain = output act various lengths / output act various lengths = 1
#
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


def main(model, base_results_dir, data_set_params, cont_int_scale, top_n=50, n_channels=64,
         top_n_per_channel_trackers=None):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    bubble_tile_size = np.array([7, 7])
    c_len_bins = [25, 50, 75, 100, 125, 150, 175, 200]

    required_data_set_params = [
        'biped_dataset_dir', 'biped_dataset_type', 'n_biped_imgs', 'n_epochs']
    for param in required_data_set_params:
        if param not in data_set_params:
            raise Exception("Required Dataset Param {} not found".format(param))

    # Immutable
    # ---------
    np.set_printoptions(precision=3)

    total_n_imgs = data_set_params['n_biped_imgs'] * data_set_params['n_epochs']

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # Results folder: structure
    #   experiment_gain_vs_len_natural_images
    #       individual_channels
    #           io_gains
    #           oo_gains
    #           activations
    #           predictions
    results_dir = os.path.join(
        base_results_dir, 'experiment_gain_vs_len_natural_images_test')
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
    if top_n_per_channel_trackers is None:
        print("Finding Optimal Stimuli for each Channel. Num Images {} ...".format(total_n_imgs))
        top_n_per_channel_trackers = find_best_stimuli_for_each_channel(
            model, data_loader, top_n, n_channels, ch_mean, ch_std,
            data_set_params['n_epochs'], cont_int_scale)

    # -----------------------------------------------------------------------------------
    # Effect of contour length
    # -----------------------------------------------------------------------------------
    print("Finding responses to contours of different length ...")

    summary_file = os.path.join(results_dir, 'summary.txt')
    f_handle = open(summary_file, 'w+')

    f_handle.write("Settings {}\n".format('-' * 80))
    f_handle.write("Bubble Tile Size {}\n".format(bubble_tile_size[0]))
    f_handle.write("Contour Lengths Considered {} \n".format(c_len_bins))
    f_handle.write("Results {}\n".format('-' * 80))

    # Variables to track across all channels
    mean_out_acts = np.ones((n_channels, len(c_len_bins))) * -1000
    std_out_acts = np.ones_like(mean_out_acts) * -1000
    mean_in_acts = np.ones_like(mean_out_acts) * -1000
    std_in_acts = np.ones_like(mean_out_acts) * -1000
    mean_preds = np.ones_like(mean_out_acts) * -1000
    std_preds = np.ones_like(mean_out_acts) * -1000
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
        tgt_n_in_act_mat = np.ones((n_images, len(c_len_bins))) * -1000
        tgt_n_out_act_mat = np.ones_like(tgt_n_in_act_mat) * -1000
        tgt_n_pred_mat = np.ones_like(tgt_n_in_act_mat) * -1000  # invalid value

        # (in, out) activations for image with continuous contour
        tgt_n_cont_c_acts = np.zeros((n_images, 2))

        for item_idx, item in enumerate(max_active_nodes):
            # Find the closest point on contour to max active point
            d_to_contour = \
                data_set.get_distance_point_and_contour(
                    np.array(item.position) * cont_int_scale, item.c1)

            closest_contour_point_idx = np.argmin(d_to_contour)

            tgt_n_cont_c_acts[item_idx, ] = np.array([item.in_act, item.out_act])

            for c_len_idx, c_len in enumerate(c_len_bins):

                c1 = item.c1.numpy()

                ep1_idx = closest_contour_point_idx + c_len // 2
                ep2_idx = closest_contour_point_idx - c_len // 2

                img = data_set.get_img_by_index(item.index)

                plt.figure()
                plt.imshow(np.transpose(img, axes=(1, 2, 0)))
                plt.scatter(item.c1[:, 1], item.c1[:, 0], marker='.', color='magenta',
                            label='contour')
                plt.title('Original Image')

                # Put bubbles at all locations not in the selected contour
                if ep2_idx < ep1_idx:

                    visible_c1 = c1[ep2_idx:ep1_idx]
                    visible_idxs = np.arange(ep2_idx, ep1_idx)
                    deleted_c1 = np.delete(c1, visible_idxs, axis=0)
                else:
                    visible_c1 = c1[ep1_idx:ep2_idx]
                    visible_idxs = np.arange(ep1_idx, ep2_idx)
                    deleted_c1 = np.delete(c1, visible_idxs, axis=0)

                img = data_set.get_img_by_index(item.index)

                puncture = utils.PunctureImage(
                    n_bubbles=1, fwhm=bubble_tile_size[0], tile_size=bubble_tile_size)

                bubble_insert_locations = deleted_c1 - bubble_tile_size // 2
                punctured_img = puncture(
                    img,
                    start_loc_arr=bubble_insert_locations)

                data_set.add_end_stop(punctured_img, c1[ep1_idx])
                data_set.add_end_stop(punctured_img, c1[ep2_idx])

                plt.figure()
                plt.imshow(np.transpose(punctured_img, axes=(1, 2, 0)))
                # plt.scatter(visible_c1[:, 1], visible_c1[:, 0], marker='.', color='magenta',
                #             label='contour')
                plt.title('Punctured image Image')

                import pdb
                pdb.set_trace()








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
        'n_biped_imgs': 100,
        'n_epochs': 1  # Total images = n_epochs * n_biped_images
    }

    # Model
    # ------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)

    net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    saved_model = \
        './results/pathfinder/' \
        'BinaryClassifierResnet50_CurrentSubtractInhibitLayer_20200811_094315_no_max_pooling/' \
        'best_accuracy.pth'
    scale_down_input_to_contour_integration_layer = 2

    # # Control Model
    # # -----
    # cont_int_layer = new_control_models.ControlMatchParametersLayer(
    #     lateral_e_size=15, lateral_i_size=15)
    # # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    # #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # net = new_piech_models.BinaryClassifierResnet50(cont_int_layer)
    # saved_model = \
    #     './results/pathfinder/' \
    #     'BinaryClassifierResnet50_ControlMatchParametersLayer_20200807_214527_base/' \
    #     'best_accuracy.pth'
    # scale_down_input_to_contour_integration_layer = 4

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

