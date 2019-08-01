#  --------------------------------------------------------------------------------------
#   Pytorch Generator for the Fields 1993 Dataset
#  --------------------------------------------------------------------------------------
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as transform_functional
from torch.utils.data import DataLoader, Dataset


class Fields1993(Dataset):
    def __init__(self, data_dir, augment=False, transform=None):
        self.data_dir = data_dir
        self.augment = augment
        self.transform = transform

        image_dir = os.path.join(self.data_dir, 'images')
        label_dir = os.path.join(self.data_dir, 'labels')
        data_key = os.path.join(self.data_dir, 'data_key.txt')

        if not os.path.exists(self.data_dir):
            raise Exception("Base directory {} not found!".format(self.data_dir))
        if not os.path.exists(image_dir):
            raise Exception("Image Directory {} not found!".format(image_dir))
        if not os.path.exists(label_dir):
            raise Exception("Label Directory {} not found!".format(label_dir))
        if not os.path.exists(data_key):
            raise Exception("Data Key {} Not Found!".format(data_key))

        # get handles to all files in the dataset
        with open(data_key, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.labels = [os.path.join(label_dir, x + ".npy") for x in file_names]

        assert (len(self.images) == len(self.labels))

        print("DataSet Contains {} Images".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        img = transform_functional.to_tensor(img)

        target = np.load(self.labels[index])
        target = torch.from_numpy(np.array(target)).unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    batch_size = 10

    # -------------------------------------------
    print("Setting up the Train Data Loaders")
    train_set = Fields1993(data_dir="./data/curved_contours/train")

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False
    )

    train_generator = training_data_loader.__iter__()
    train_imgs, train_labels = train_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))

    print("Setting up the Test Data Loader")
    test_set = Fields1993(data_dir='./data/curved_contours/test')

    test_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False
    )

    test_generator = test_data_loader.__iter__()
    test_imgs, test_labels = test_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))










