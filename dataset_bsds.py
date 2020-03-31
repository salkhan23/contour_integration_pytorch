# ---------------------------------------------------------------------------------------
# Pytorch Data set/loaders for the BSDS Segmentation Task
# ---------------------------------------------------------------------------------------
import os
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as transform_functional

from PIL import Image


class BSDS(data.Dataset):
    """
    Berkeley Segmentation Data Set with figure-ground labels.

    param: data_dir (string): Root directory of the VOC Dataset.
    param: image_set (string, optional): Select the image_set to use, ``train`` or ``test``

    """

    def __init__(self, data_dir, image_set='train', augment=False, transform=None):
        self.data_dir = data_dir
        self.image_set = image_set
        self.augment = augment
        self.transform = transform

        self.key_file = os.path.join(self.data_dir,  self.image_set + "_pair.lst")
        # print(self.key_file)
        if not os.path.exists(self.key_file):
            raise Exception("Cannot find mapping file: {}".format(self.key_file))

        self.images = []
        self.labels = []

        with open(self.key_file, 'r') as h:
            file_strings = h.readlines()

        for line in file_strings:
            self.images.append(os.path.join(data_dir, line.split()[0]))
            self.labels.append(os.path.join(data_dir, line.split()[1]))

        self.resize = transforms.Resize(size=(256, 256))

        self.data_set_mean, self.data_set_std = self.get_data_set_mean_and_std()
        print("DataSet Contains {} Images.\nChannel mean {},\nChannel std {}".format(
            len(self.images), self.data_set_mean, self.data_set_std))

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

        img = self.resize(img)
        img = transform_functional.to_tensor(img)

        # Dont use the convert to [0, 1] leave the intensities as is.
        # Also all the channels are the same so just pick the first one
        # target = Image.open(self.labels[index]).convert("RGB")
        # target = transform_functional.to_tensor(target)
        # target = torch.unsqueeze(target[0, ], dim=0)

        target = Image.open(self.labels[index]).convert("L")
        target = self.resize(target)
        target = transform_functional.to_tensor(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data_set_mean_and_std(self):
        """
        Compute the data set mean and standard deviation
        Var[x] = E[X ^ 2] - E ^ 2[X]
        Ref: https://discuss.pytorch.org/
             t/about-normalization-using-pre-trained-vgg16-networks/23560/9
        """
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for idx in range(self.__len__()):
            img, _ = self.__getitem__(idx)

            c, h, w = img.shape
            nb_pixels = h * w
            sum_ = torch.sum(img, dim=[1, 2])
            sum_of_square = torch.sum(img ** 2, dim=[1, 2])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.ion()

    data_set_dir = './data/bsds'

    plt.ion()
    batch_size = 1
    n_steps = 100

    # Imagenet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    print("Setting up the Train Data Loaders")

    train_set = BSDS(data_dir=data_set_dir, image_set='train', transform=normalize)

    training_data_loader = data.DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    train_generator = training_data_loader.__iter__()

    for s_idx in range(n_steps):

        train_imgs, train_labels = train_generator.__next__()

        print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))
        print("Batch mean = {}. std = {}".format(
            torch.mean(train_imgs, dim=[0, 2, 3]), torch.std(train_imgs, dim=[0, 2, 3])))

        display_img_idx = 0
        display_img = train_imgs[display_img_idx, ]
        display_label = train_labels[display_img_idx, ]

        f, ax_arr = plt.subplots(1, 2)
        ax_arr[0].imshow(np.transpose(display_img, axes=(1, 2, 0)))
        ax_arr[1].imshow(np.squeeze(display_label))

        import pdb
        pdb.set_trace()

        plt.close(f)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
