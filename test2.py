import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from models.piech_models import CurrentSubtractiveInhibition
import dataset
import utils

if __name__ == "__main__":
    plt.ion()
    # -----------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CurrentSubtractiveInhibition().to(device)

    saved_model = './results/CurrentSubtractiveInhibition_20190827_192404/trained_epochs_50.pth'

    # Data loaders
    # -----------------------------------------------------------------------------------
    print("====> Setting up data loaders ")
    data_set_dir = "./data/fields_2006_contour_length"
    print("Source: {}".format(data_set_dir))

    # get mean/std of dataset
    meta_data_file = os.path.join(data_set_dir, 'dataset_metadata.pickle')
    with open(meta_data_file, 'rb') as handle:
        meta_data = pickle.load(handle)
    print("Channel mean {}, std {}".format(meta_data['channel_mean'], meta_data['channel_std']))

    # Pre-processing
    normalize = transforms.Normalize(
        mean=meta_data['channel_mean'],
        std=meta_data['channel_std']
    )

    data_dir = os.path.join(data_set_dir, 'test')
    bg_tile_size = meta_data["full_tile_size"]

    c_len_1_data_set = dataset.Fields1993(
        data_dir=data_dir, bg_tile_size=bg_tile_size, transform=normalize, c_len_arr=[1])
    c_len_1_data_loader = DataLoader(
        dataset=c_len_1_data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

    c_len_3_data_set = dataset.Fields1993(
        data_dir=data_dir, bg_tile_size=bg_tile_size, transform=normalize, c_len_arr=[3])
    c_len_3_data_loader = DataLoader(
        dataset=c_len_3_data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

    c_len_5_data_set = dataset.Fields1993(
        data_dir=data_dir, bg_tile_size=bg_tile_size, transform=normalize, c_len_arr=[5])
    c_len_5_data_loader = DataLoader(
        dataset=c_len_5_data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

    c_len_7_data_set = dataset.Fields1993(
        data_dir=data_dir, bg_tile_size=bg_tile_size, transform=normalize, c_len_arr=[7])
    c_len_7_data_loader = DataLoader(
        dataset=c_len_7_data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

    c_len_9_data_set = dataset.Fields1993(
        data_dir=data_dir, bg_tile_size=bg_tile_size, transform=normalize, c_len_arr=[9])
    c_len_9_data_loader = DataLoader(
        dataset=c_len_9_data_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)

    # ---------------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss().to(device)
    detect_thres = 0.5

    # -----------------------------------------------------------------------------------
    def validate(test_data_loader):
        """ Get loss over validation set """
        model.eval()
        e_loss = 0
        e_iou = 0

        with torch.no_grad():
            for iteration, (img, label) in enumerate(test_data_loader, 1):
                img = img.to(device)
                label = label.to(device)

                label_out, iter_pred_out, iter_act_out = model(img)
                batch_loss = criterion(label_out, label.float())

                center_neuron_excitation_list = []
                center_neuron_inhibition_list = []

                for idx, (excite_act, inhibit_act) in enumerate(iter_act_out):

                    center_neuron_excite = excite_act[:, :, excite_act.shape[2] // 2, excite_act.shape[3] // 2]
                    center_neuron_inhibit = inhibit_act[:, :, inhibit_act.shape[2] // 2, inhibit_act.shape[3] // 2]

                    center_neuron_excite = center_neuron_excite.cpu().detach().numpy()
                    center_neuron_inhibit = center_neuron_inhibit.cpu().detach().numpy()

                    center_neuron_excite = np.squeeze(center_neuron_excite, axis=0)
                    center_neuron_inhibit = np.squeeze(center_neuron_inhibit, axis=0)

                    center_neuron_excitation_list.append(center_neuron_excite)
                    center_neuron_inhibition_list.append(center_neuron_inhibit)

                center_neuron_excitation_list = np.array(center_neuron_excitation_list)
                center_neuron_inhibition_list = np.array(center_neuron_inhibition_list)

                f, ax_arr = plt.subplots(2, 1)
                for idx in range(center_neuron_excitation_list.shape[0]):
                    ax_arr[0].plot(center_neuron_excitation_list[idx, ], label='iter_{}'.format(idx))
                    ax_arr[1].plot(center_neuron_inhibition_list[idx, ], label='iter_{}'.format(idx))

                ax_arr[0].set_title("Excitation")
                ax_arr[0].grid()
                ax_arr[1].set_title("Inhibition")
                ax_arr[1].grid()
                plt.legend()

                break
                import pdb
                pdb.set_trace()
                # plt.close(f)




                e_loss += batch_loss.item()
                preds = (label_out > detect_thres)
                e_iou += utils.intersection_over_union(preds.float(), label.float()).cpu().detach().numpy()

        e_loss = e_loss / len(test_data_loader)
        e_iou = e_iou / len(test_data_loader)

        # print("Val Loss = {:0.4f}, IoU={:0.4f}".format(e_loss, e_iou))

        return e_loss, e_iou

    validate(c_len_1_data_loader)
    plt.suptitle("c_len 1")
    validate(c_len_3_data_loader)
    plt.suptitle("c_len 3")
    validate(c_len_5_data_loader)
    plt.suptitle("c_len 5")
    validate(c_len_7_data_loader)
    plt.suptitle("c_len 7")
    validate(c_len_9_data_loader)
    plt.suptitle("c_len 9")

    import pdb
    pdb.set_trace()

    input("Press any Key to Exit")
