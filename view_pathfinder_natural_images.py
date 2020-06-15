# ---------------------------------------------------------------------------------------
# View an already generated pathfinder dataset
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader, Dataset


class PathfinderNaturalImages(Dataset):

    def __init__(self, data_dir, transforms=None):

        self.transforms = transforms

        imgs_dir = os.path.join(data_dir, 'images')
        labels_file = os.path.join(data_dir, 'classification_labels.txt')
        data_key_file = os.path.join(data_dir, 'data_key.txt')

        required_items = [imgs_dir, labels_file, data_key_file]
        for item in required_items:
            if not os.path.exists(item):
                raise Exception('Required item {} not found'.format(item))

        debug_indv_contours_labels_dir = os.path.join(data_dir, 'individual_contours_labels')
        debug_full_labels_dir = os.path.join(data_dir, 'full_labels')

        debug_distance_between_points = os.path.join(data_dir, 'distances_between_points.txt')
        debug_map_to_org_imgs = os.path.join(data_dir, 'map_to_original_images.txt')

        debug_items = [
            debug_indv_contours_labels_dir,
            debug_full_labels_dir,
            debug_distance_between_points,
            debug_map_to_org_imgs
        ]
        for item in debug_items:
            if not os.path.exists(item):
                raise Exception('Debug item {} not found'.format(item))

        with open(data_key_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(imgs_dir, file) for file in file_names]
        self.indv_contours_labels = \
            [os.path.join(debug_indv_contours_labels_dir, file) for file in file_names]
        self.full_labels = [os.path.join(debug_full_labels_dir, file) for file in file_names]

        with open(labels_file, "r") as f:
            self.labels = [int(x.strip()) for x in f.readlines()]

        with open(debug_distance_between_points, "r") as f:
            self.distances = [int(x.strip()) for x in f.readlines()]

        with open(debug_map_to_org_imgs, "r") as f:
            self.org_imgs = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = transform_functional.to_tensor(img)

        label = self.labels[index]
        label = torch.tensor(label)

        individual_contours_label = Image.open(self.indv_contours_labels[index]).convert('L')
        individual_contours_label = transform_functional.to_tensor(individual_contours_label)

        full_label = Image.open(self.full_labels[index]).convert('L')
        full_label = transform_functional.to_tensor(full_label)

        distance = self.distances[index]
        distance = torch.tensor(distance)

        org_img_name = self.org_imgs[index]

        return img, label, individual_contours_label, full_label, distance, org_img_name


if __name__ == "__main__":

    dataset_dir = './data/pathfinder_natural_images'
    random_seed = 5

    # Immutable ----------------------
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Data loader
    # -----------------------------------------------------------------------------------
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]

    # Pre-processing
    transforms_list = [
        transforms.Normalize(mean=ch_mean, std=ch_std)
    ]
    pre_process_transforms = transforms.Compose(transforms_list)

    dataset = PathfinderNaturalImages(dataset_dir, transforms=pre_process_transforms)

    data_loader = DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,  # Do not change for correct image mapping
        pin_memory=True
    )

    # -----------------------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------------------
    for iteration, data_loader_out in enumerate(data_loader, 0):

        imgs, labels, individual_contours_labels, full_labels, distances, org_img_names = \
            data_loader_out

        idx = 0

        img = imgs[idx, ]
        class_label = labels[idx, ]
        single_contour_label = individual_contours_labels[idx, ]
        full_label = full_labels[idx, ]
        distance = distances[idx, ]
        org_img_name = org_img_names[0]

        img = np.transpose(img, axes=(1, 2, 0))
        display_img = (img - img.min()) / (img.max() - img.min())

        f, ax_arr = plt.subplots(1, 3, figsize=(15, 5))
        ax_arr[0].imshow(display_img)
        ax_arr[1].imshow(np.squeeze(single_contour_label))
        ax_arr[1].set_title("Connected {}.".format(bool(class_label)))
        ax_arr[2].imshow(np.squeeze(full_label))

        save_dir = './results/sample_images'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        f.suptitle("Distance between points {}\n Original Image {}".format(
            distance, org_img_name))

        f.savefig(os.path.join(save_dir, 'img_{}.png'.format(iteration)))

        # import pdb
        # pdb.set_trace()

        plt.close(f)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print('End')
    import pdb
    pdb.set_trace()
