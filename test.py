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


def process_images(model, device_to_use, ch_mus, ch_sigmas, in_imgs):
    # Zero all collected variables
    global edge_extract_act
    global cont_int_in_act
    global cont_int_out_act

    edge_extract_act = 0
    cont_int_in_act = 0
    cont_int_out_act = 0

    normalize = transforms.Normalize(mean=ch_mus, std=ch_sigmas)
    model_in_img = normalize(in_imgs.squeeze())
    model_in_img = model_in_img.to(device_to_use).unsqueeze(0)

    # Pass the images through the model
    model.eval()
    if isinstance(model, new_piech_models.JointPathfinderContourResnet50):
        # Output is contour_dataset_out, pathfinder_out
        _, label_out = model(model_in_img)
    else:
        label_out = model(model_in_img)

    labels_out = torch.sigmoid(label_out)

    return label_out


class MaxActiveElement:
    """ An element in the Top-n max Active Dictionary.
        Defines what is stored about an image
    """

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

    def push(self, value, item):

        if len(self._heap) < self.depth:
            heapq.heappush(self._heap, (value, item))
        else:
            min_stored_val, min_stored_item = self.pop()  # pop the lowest

            if value > min_stored_val:
                heapq.heappush(self._heap, (value, item))

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
            v, item = self.pop()
            lst_items.append(item)
            lst_values.append(v)

        # reverse the order, max activation out first
        zipped_object = zip(lst_values, lst_items)
        zipped_object = sorted(zipped_object, reverse=True)

        unzipped_object = zip(*zipped_object)
        lst_values, lst_items = list(unzipped_object)

        return lst_items, lst_values


def main(model, results_dir):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    n_channels = 64
    top_n = 10

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    batch_size = 32

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    biped_dataset_dir = './data/BIPED/edges'
    biped_dataset_type = 'train'
    n_biped_imgs = 100

    data_set = OnlineNaturalImagesPathfinder(
        data_dir=biped_dataset_dir,
        dataset_type=biped_dataset_type,
        transform=None,
        subset_size=n_biped_imgs,
        resize_size=(256, 256),
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    for ch_idx in np.arange(n_channels):

        top_act_tracker = TopNTracker(top_n)
        valid_img_count = 0

        ch_store_dir = os.path.join(results_dir, 'channel_{}'.format(ch_idx))
        if not os.path.exists(ch_store_dir):
            os.makedirs(ch_store_dir)

        for iteration, data_loader_out in enumerate(data_loader, 1):

            # Find the top n max responsive images for target channel
            imgs, labels, sep_c_labels, full_labels, dist_arr, org_img_idx_arr, \
                c1_arr, c2_arr, start_point_arr, end_point_arr = data_loader_out

            labels_out = process_images(model, dev, ch_mean, ch_std, imgs)

            for img_idx in range(labels_out.shape[0]):

                # Get the target img and labels
                img = imgs[img_idx, ]
                label = labels[img_idx, ]
                sep_c_label = sep_c_labels[img_idx, ]
                full_label = full_labels[img_idx, ]
                d = dist_arr[img_idx, ]
                biped_img_idx = org_img_idx_arr[img_idx, ]
                c1 = c1_arr[img_idx, ]
                c2 = c2_arr[img_idx, ]
                start_point = start_point_arr[img_idx, ]
                end_point = end_point_arr[img_idx, ]

                curr_tgt_ch_acts = cont_int_out_act[img_idx, ch_idx, :, :]
                curr_max_act = np.max(curr_tgt_ch_acts)
                curr_max_act_idx = np.argmax(curr_tgt_ch_acts)  # 1d index
                curr_max_act_idx = \
                    np.unravel_index(curr_max_act_idx, curr_tgt_ch_acts.shape)  # 2d idx

                # Check for valid input images
                # 1. Endpoints should be connected
                # 2. max_active should be at most one pixel away from the contour

                if label:
                    # The max active neuron is on the contour (or very close by)
                    # the / 4 is to get the position of the contour @ the scale of the
                    # contour integration activation
                    min_d_to_contour = np.min(
                        data_set.get_distance_point_and_contour(curr_max_act_idx, c1 // 4))

                    if min_d_to_contour < 1.5:
                        node = MaxActiveElement(
                            activation=curr_max_act,
                            position=curr_max_act_idx,
                            index=biped_img_idx,
                            c1=c1,
                            c2=c2,
                            ep1=start_point,  # get rid of batch dim
                            ep2=end_point,
                            prediction=labels_out[img_idx].item(),
                            gt=label
                        )

                        top_act_tracker.push(curr_max_act, node)
                        valid_img_count += 1
                        print("Adding image to stored images")

        max_active_nodes, values = top_act_tracker.get_stored_values()
        print("Number of Good images found {}. Stored top {}".format(
            valid_img_count, len(max_active_nodes)))


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

    # results_store_dir = os.path.dirname(saved_model)
    results_store_dir = './results/sample_images'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(saved_model, map_location=device))
    main(net, results_store_dir)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb

    pdb.set_trace()
