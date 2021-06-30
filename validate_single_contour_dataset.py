# ---------------------------------------------------------------------------------------
#  Generate predictions of a trained model on the single contours dataset.
#  Dataset has be be previously created, see  generate_data/generate_single_contour_dataset
#  Data set structure:
#        main
#           contour_len_bin_25-49
#           contour_len_bin_50-99
#           ---
#
#       Each sub directory should have 2 folders,
#            images  (full images)
#            label (highlighted single contour)
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader, Dataset

from models import new_piech_models
from models import new_control_models


class SingleContourDataSet(Dataset):
    def __init__(self, data_dir, c_len_sub_dir, transform=None):

        self.transform = transform

        if not os.path.exists(data_dir):
            raise Exception("Cannot find data dir {}".format(data_dir))
        self.data_dir = data_dir

        img_dir = os.path.join(data_dir, 'images', c_len_sub_dir)
        label_dir = os.path.join(data_dir, 'labels', c_len_sub_dir)

        if not os.path.exists(img_dir):
            raise Exception("Images dir {} DNE".format(img_dir))
        if not os.path.exists(label_dir):
            raise Exception("Labels dir {} DNE".format(label_dir))

        list_of_files = os.listdir(img_dir)
        self.images = [os.path.join(img_dir, file) for file in list_of_files]
        self.labels = \
            [os.path.join(label_dir, file.replace('.jpg', '.png')) for file in list_of_files]

        print("Dataset contains {} Images".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img1 = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.labels[index]).convert('L')  # Greyscale

        img1 = transform_functional.to_tensor(img1)
        target = transform_functional.to_tensor(target)

        # todo: Figure out why this weird scaling is happening
        target = (target - target.min()) / (target.max() - target.min())

        if self.transform is not None:
            img1 = self.transform(img1)

        return img1, target


def get_predictions(model, data_dir, results_dir):
    """

    :param model:
    :param data_dir:
    :param results_dir:
    :return:
    """

    list_of_sub_dirs = os.listdir(os.path.join(data_dir, 'labels'))  # sub dirs of contour lengths
    list_of_sub_dirs.sort(key=lambda x1: float(x1.split('_')[1]))

    predicts_base_dir = os.path.join(results_dir, 'predictions_' + data_dir.split('/')[-1] + '_test')

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sb_dir_idx, sub_dir in enumerate(list_of_sub_dirs):

        print("Processing contours of length {} ...".format(sub_dir))

        preds_dir = os.path.join(predicts_base_dir, sub_dir)
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        # print('storing results @ {}'.format(preds_store_dir))

        # -------------------------------------------------------------------------------
        # Data Loader
        # -------------------------------------------------------------------------------
        print("Setting up Dataloader")
        # Imagenet Mean and STD
        ch_mean = [0.485, 0.456, 0.406]
        ch_std = [0.229, 0.224, 0.225]

        # Pre-processing
        pre_process_transforms = transforms.Compose([
            transforms.Normalize(mean=ch_mean, std=ch_std),
            # utils.PunctureImage(n_bubbles=100, fwhm=20, peak_bubble_transparency=0)
        ])

        dataset = SingleContourDataSet(data_dir, sub_dir, pre_process_transforms)
        data_loader = DataLoader(
            dataset=dataset,
            num_workers=4,
            batch_size=1,
            shuffle=False,  # Do not change for correct image mapping
            pin_memory=True
        )

        # -------------------------------------------------------------------------------
        # Main Loop
        # -------------------------------------------------------------------------------
        model.eval()

        list_of_labels = data_loader.dataset.labels

        with torch.no_grad():
            for iteration, (img, label) in enumerate(data_loader, 0):
                img = img.to(dev)
                label = label.to(dev)

                label_out = model(img)

                # Before visualizing Sigmoid the output.
                # This is already done in the loss function
                label_out = torch.sigmoid(label_out)
                label_out = label * label_out  # To get only pixels of interest (single contour)
                label_out = label_out.cpu().detach().numpy()
                label_out = np.squeeze(label_out)

                plt.imsave(
                    fname=os.path.join(preds_dir, list_of_labels[iteration].split('/')[-1]),
                    arr=np.squeeze(label_out),
                    cmap=plt.cm.gray,
                )

                # # Plot Input image, label and prediction
                # img = img.detach().cpu().numpy()
                # img = np.squeeze(img)
                #
                # label = label.detach().cpu().numpy()
                # label = np.squeeze(label)
                #
                # # Debug Plot
                # f, ax_arr = plt.subplots(1, 3)
                #
                # img = np.transpose(img, axes=[1, 2, 0])
                # display_img = (img - img.min()) / (img.max() - img.min())
                #
                # ax_arr[0].imshow(display_img)
                # ax_arr[0].set_title("Image")
                #
                # ax_arr[1].imshow(label)
                # ax_arr[1].set_title("GT")
                #
                # ax_arr[2].imshow(label_out)
                # ax_arr[2].set_title("Model Corresponding Prediction")
                #
                # import pdb
                # pdb.set_trace()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    data_set_dir = './data/single_contour_natural_images_4'
    random_seed = 5

    # Build Model
    # cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)
    # cont_int_layer = new_piech_models.CurrentDivisiveInhibitLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5, use_recurrent_batch_norm=True)

    cont_int_layer = new_control_models.ControlMatchParametersLayer(
        lateral_e_size=15, lateral_i_size=15)
    # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    #     lateral_e_size=15, lateral_i_size=15, n_iters=5)

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

    # saved_model = \
    #     './results/biped' \
    #     '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base' \
    #     '/last_epoch.pth'

    saved_model = \
        'results/biped' \
        '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base' \
        '/last_epoch.pth'

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

    # Immutable
    # ------------------------------------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    net.load_state_dict(torch.load(saved_model, map_location=device))

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    get_predictions(
        net,
        data_set_dir,
        os.path.dirname(saved_model))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    input("Press any Key to continue")
    pdb.set_trace()
