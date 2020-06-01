# ---------------------------------------------------------------------------------------
# Get performance of a Trained Edge Extraction Model over the Single Natural Image Contour
# Dataset
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
    def __init__(self, data_dir,  transform=None):

        self.transform = transform

        if not os.path.exists(data_dir):
            raise Exception("Cannot find data dir {}".format(data_dir))
        self.data_dir = data_dir

        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

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

        # todo: What is this weird scaling
        target[target > 0.2] = 1
        target[target <= 0.2] = 0

        if self.transform is not None:
            img1 = self.transform(img1)

        return img1, target


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    data_set_dir = './data/single_contour_natural_images'
    random_seed = 5

    save_predictions = True

    # # # Control Model
    # # ----------------
    # # cont_int_layer = new_control_models.ControlMatchIterationsLayer(
    # #     lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # cont_int_layer = new_control_models.ControlMatchParametersLayer(
    #     lateral_e_size=15, lateral_i_size=15)
    # saved_model = \
    #     './results/biped' \
    #     '/EdgeDetectionResnet50_ControlMatchParametersLayer_20200508_001539_base' \
    #     '/last_epoch.pth'

    # Model
    # -----
    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    saved_model = \
        './results/biped' \
        '/EdgeDetectionResnet50_CurrentSubtractInhibitLayer_20200430_131825_base' \
        '/last_epoch.pth'

    net = new_piech_models.EdgeDetectionResnet50(cont_int_layer)

    # immutable
    # ------------------------------------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net.to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))

    list_of_sub_dirs = os.listdir(data_set_dir)
    list_of_sub_dirs.sort(key=lambda x: float(x.strip('len_')))

    for sb_dir_idx, sub_dir in enumerate(list_of_sub_dirs):

        print("Processing sub directory {} ...".format(sub_dir))

        # Create the predictions store directory
        base_store_dir = os.path.dirname(saved_model)
        preds_dir = os.path.join(
            base_store_dir, 'predictions_single_contour_dataset', sub_dir)
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

        dataset = SingleContourDataSet(os.path.join(data_set_dir, sub_dir), pre_process_transforms)
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
        net.eval()

        list_of_labels = data_loader.dataset.labels

        with torch.no_grad():
            for iteration, (img, label) in enumerate(data_loader, 0):
                img = img.to(device)
                label = label.to(device)

                label_out = net(img)

                # Before visualizing Sigmoid the output.
                # This is already done in the loss function
                label_out = torch.sigmoid(label_out)
                label_out = label * label_out
                label_out = label_out.cpu().detach().numpy()

                if save_predictions:
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
                # label_out = label_out.detach().cpu().numpy()
                # label_out = np.squeeze(label_out)
                #
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

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    input("Press any Key to continue")
    pdb.set_trace()
