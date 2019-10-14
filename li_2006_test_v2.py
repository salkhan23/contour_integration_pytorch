import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models.new_piech_models as new_piech_models
import dataset
import utils


contour_integration_in_act = 0
contour_integration_out_act = 0


def contour_integration_cb(self, layer_in, layer_out):
    """ Attach at Contour Integration layer"""
    global contour_integration_in_act
    global contour_integration_out_act

    contour_integration_in_act = layer_in[0].cpu().detach().numpy()
    contour_integration_out_act = layer_out.cpu().detach().numpy()


def get_tgt_neuron_acts(model, tgt_neuron, data_loader):

    global contour_integration_in_act
    global contour_integration_out_act

    model.eval()
    detect_thresh = 0.5
    e_iou = 0

    tgt_neuron_in_acts = []
    tgt_neuron_out_acts = []

    with torch.no_grad():
        for iteration, (img, label) in enumerate(data_loader, 0):
            img = img.to(device)
            label = label.to(device)

            # Zero All activations before each iteration
            contour_integration_in_act = 0
            contour_integration_out_act = 0

            label_out = model(img)
            preds = (label_out > detect_thresh)

            e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

            # Store the activations
            _, _, r, c = contour_integration_in_act.shape
            tgt_neuron_in_acts.append(contour_integration_in_act[0, tgt_neuron, r // 2, c // 2])
            tgt_neuron_out_acts.append(contour_integration_out_act[0, tgt_neuron, r // 2, c // 2])

            # # show the image
            # import pdb
            # pdb.set_trace()

    e_iou = e_iou / len(data_loader)

    return e_iou, np.array(tgt_neuron_in_acts), np.array(tgt_neuron_out_acts)


def get_contour_gain_vs_length(
        model, c_len_arr, beta_arr, data_folder, params_idx, full_tile_size, tgt_neuron, epsilon=1e-5):

    iou_scores = []
    cont_int_gain_means = []
    cont_int_gain_stds = []

    # Common
    # ----------------------
    mean = meta_data['channel_mean']
    mean = np.array([0.36167968, 0.3632432,  0.36181496])
    std = meta_data['channel_std']
    std = np.array([0.43319716, 0.43500246, 0.43306438])

    normalize = transforms.Normalize(mean=mean, std=std)

    for c_idx, c_len in enumerate(c_len_arr):
        # Data Loader
        # ----------
        data_set = dataset.Fields1993(
            data_dir=data_folder,
            bg_tile_size=full_tile_size,
            transform=normalize,
            c_len_arr=[c_len],
            beta_arr=beta_arr,
            gabor_set_arr=[params_idx]
        )

        data_loader = DataLoader(data_set, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

        avg_iou, tgt_n_in_arr, tgt_n_out_arr = get_tgt_neuron_acts(model, tgt_neuron, data_loader)

        iou_scores.append(avg_iou)

        cont_int_gain_arr = tgt_n_out_arr / (tgt_n_in_arr + epsilon)

        # plt.figure()
        # plt.plot(cont_int_gain_arr)
        #
        # import pdb
        # pdb.set_trace()

        cont_int_gain_means.append(cont_int_gain_arr.mean())
        cont_int_gain_stds.append(cont_int_gain_arr.std())

    return iou_scores, np.array(cont_int_gain_means), np.array(cont_int_gain_stds)


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    frag_size = np.array([7, 7])

    image_size = np.array([224, 224, 3])

    plt.ion()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("====> Setting up the Model ...")
    net = new_piech_models.ContourIntegrationCSI(lateral_e_size=23, lateral_i_size=23)
    saved_model = './results/new_model/ContourIntegrationCSI_20191005_195738_base/best_accuracy.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model))

    # Register Callback
    net.contour_integration_layer.register_forward_hook(contour_integration_cb)

    # -----------------------------------------------------------------------------------
    # Data Source
    # -----------------------------------------------------------------------------------
    print("====> Data Initialization")

    # In this data set an optimal stimulus is defined for each channel
    data_set_dir = "./data/test"
    print("Source: {}".format(data_set_dir))

    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)

    data_dir = os.path.join(data_set_dir, 'val')
    n_images_per_set = meta_data['n_val_images_per_set']
    n_channels = 64

    bg_tile_size = meta_data["full_tile_size"]

    # -----------------------------------------------------------------------------------
    # Main Routine
    # -----------------------------------------------------------------------------------
    contour_len_arr = [1, 3, 5, 7, 9, 12]
    beta_rotation_arr = [0]

    for gp_idx, gabor_params in enumerate(meta_data['g_params_list']):

        # gp_idx = 11
        # gabor_params = meta_data['g_params_list'][11]

        print("Processing Gabor Param set {}".format(gp_idx))

        ious, mean_gains, std_gains = get_contour_gain_vs_length(
            net,
            contour_len_arr,
            beta_rotation_arr,
            data_dir,
            gp_idx,
            full_tile_size=bg_tile_size,
            tgt_neuron=11
        )

        plt.figure()
        plt.errorbar(contour_len_arr, mean_gains, std_gains, label='Neuron {}'.format(gp_idx))
        plt.xlabel("Contour Length")
        plt.ylabel("Gain")
        plt.legend()
        # plt.title("Neuron {}. Is max active for stimuli {}, Is FF act > 15 {}".format(
        #     gp_idx, gabor_params[0]['is_max_active'], gabor_params[0]['optimal_stimulus_act'] > 15))

        import pdb
        pdb.set_trace()