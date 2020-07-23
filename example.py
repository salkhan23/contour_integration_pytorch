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


def add_contour(in_img, c):
    """ in img should be 2D. In place operation """
    for point in c:
        in_img[point[0], point[1]] = 1


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

        # reverse the order, max first.
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
    model.load_state_dict(torch.load(saved_model, map_location=dev))
    model = model.to(dev)

    model.edge_extract.register_forward_hook(edge_extract_cb)
    model.contour_integration_layer.register_forward_hook(contour_integration_cb)

    n_channels = 64

    top_n = 10

    # Imagenet Mean and STD
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # -----------------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------------
    biped_dataset_dir = './data/BIPED/edges'
    biped_dataset_type = 'train'
    n_biped_imgs = 5000

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
        batch_size=1,  # must stay as one
        shuffle=False,
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------

    for ch_idx in np.arange(n_channels):
        top_act_tracker = TopNTracker(top_n)

        ch_store_dir = os.path.join(results_dir, 'channel_{}'.format(ch_idx))
        if not os.path.exists(ch_store_dir):
            os.makedirs(ch_store_dir)

        for iteration, data_loader_out in enumerate(data_loader, 1):
            # -----------------------------------------------------------------------------
            # Find top n max responsive images for this channel
            # -----------------------------------------------------------------------------
            imgs, labels, sep_c_label, full_label, d, org_img_idx, c1, c2, \
                start_point, end_point = data_loader_out

            label_out = process_image(model, dev, ch_mean, ch_std, imgs)

            # remove batch dimension
            labels = np.squeeze(labels)
            # sep_c_label = np.squeeze(sep_c_label)
            # full_label = np.squeeze(full_label)
            # d = np.squeeze(d)
            org_img_idx = np.squeeze(org_img_idx)
            c1 = np.squeeze(c1)
            c2 = np.squeeze(c2)
            start_point = np.squeeze(start_point)
            end_point = np.squeeze(end_point)

            # Target channel activation
            curr_tgt_ch_acts = cont_int_out_act[0, ch_idx, :, :]
            curr_max_act = np.max(curr_tgt_ch_acts)

            curr_max_act_idx = np.argmax(curr_tgt_ch_acts)  # 1d index
            curr_max_act_idx = np.unravel_index(
                curr_max_act_idx, curr_tgt_ch_acts.shape)  # 2d idx

            # Check for valid sample:
            # 1. Endpoints should be connected
            # 2. max_active should be at most one pixel away from the contour

            if labels:  # points are connected

                # The max active neuron is on the contour (or very close by)
                # the / 4 is to get the position of the contour @ the scale of the
                # contour integration activation
                d_to_contour = np.min(
                    data_set.get_distance_point_and_contour(curr_max_act_idx, c1 // 4))

                if d_to_contour < 1.5:
                    node = MaxActiveElement(
                        activation=curr_max_act,
                        position=curr_max_act_idx,
                        index=org_img_idx,
                        c1=c1,
                        c2=c2,
                        ep1=start_point,  # get rid of batch dim
                        ep2=end_point,
                        prediction=label_out.item(),
                        gt=labels
                    )

                    top_act_tracker.push(curr_max_act, node)

            print("Iteration {}. Target Channel Max Act {:0.4f}".format(iteration, curr_max_act))

        max_active_nodes, values = top_act_tracker.get_stored_values()
        print("Number of Good images found {}".format(len(max_active_nodes)))

        # -----------------------------------------------------------------------------------
        # Plot Stored Results
        # -----------------------------------------------------------------------------------
        for item_idx, item in enumerate(max_active_nodes):

            new_img = data_set.get_img_by_index(item.index, item.ep1, item.ep2)

            # Process the image
            new_img = torch.unsqueeze(new_img, dim=0)
            label_out = process_image(model, dev, ch_mean, ch_std, new_img)

            # Display the image
            # -----------------
            f, ax_arr = plt.subplots(1, 2, figsize=(14, 7))

            new_img = np.transpose(new_img.squeeze(), axes=(1, 2, 0))
            ax_arr[0].imshow(new_img)

            tgt_ch_acts = cont_int_out_act[0, ch_idx, :, :]
            max_act_idx = np.argmax(tgt_ch_acts)  # 1d index
            max_act_idx = np.unravel_index(max_act_idx, tgt_ch_acts.shape)  # 2d i
            ax_arr[1].imshow(tgt_ch_acts)
            ax_arr[1].set_title("Target channel: Max {:0.2f} @ {}".format(
                np.max(tgt_ch_acts), max_act_idx))
            # flip x and y when plotting
            ax_arr[1].scatter(max_act_idx[1], max_act_idx[0], marker='+', color='red', s=120)
            ax_arr[1].scatter(item.ep1[1] // 4, item.ep1[0] // 4, marker='o', color='magenta', s=60)
            ax_arr[1].scatter(item.ep2[1] // 4, item.ep2[0] // 4, marker='o', color='magenta', s=60)
            ax_arr[1].scatter(item.c1[:, 1] // 4, item.c1[:, 0] // 4, marker='.', color='magenta')

            f.suptitle("GT = {}, prediction {:0.4f}, C1 len={}, C2 len={}".format(
                item.gt, label_out.item(), len(item.c1), len(item.c2)))

            f.savefig(os.path.join(ch_store_dir, "img_{}_connected_{}.jpg".format(
                item_idx, item.gt)), format='jpg')

            plt.close(f)

    import pdb
    pdb.set_trace()


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
    # -----------------------------------------------------------------------------------
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    net = new_piech_models.JointPathfinderContourResnet50(cont_int_layer)
    saved_model = \
        'results/joint_training/' \
        'JointPathfinderContourResnet50_CurrentSubtractInhibitLayer_20200713_230237_first_run/' \
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

    main(net, results_store_dir)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End. Script took: {} to run ".format(datetime.now() - start_time))
    import pdb
    pdb.set_trace()
