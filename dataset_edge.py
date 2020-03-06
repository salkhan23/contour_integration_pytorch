# ---------------------------------------------------------------------------------------
# Pytorch Data set/loaders for the Natural Images Edge Detection Data set
# ---------------------------------------------------------------------------------------
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class EdgeDataSet(Dataset):
    def __init__(self, data_dir, transform=None, subset_size=None):

        if not os.path.exists(data_dir):
            raise Exception("Cannot find data dir {}".format(data_dir))

        self.data_dir = data_dir

        image_dir = os.path.join(self.data_dir, 'images')
        label_dir = os.path.join(self.data_dir, 'labels')

        if not os.path.exists(image_dir):
            raise Exception("Cannot find images {}".format(image_dir))

        if not os.path.exists(label_dir):
            raise Exception("Cannot find Labels {}".format(label_dir))

        self.transform = transform

        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.labels = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]

        if subset_size is not None:
            assert subset_size < len(self.images), 'subset size {} is greater than dataset size {}'.format(
                subset_size, len(self.images))

            use_idxs = np.random.choice(np.arange(len(self.images)), size=subset_size, replace=False)
            self.images = [self.images[idx] for idx in use_idxs]
            self.labels = [self.labels[idx] for idx in use_idxs]

        if len(self.images) != len(self.labels):
            raise Exception("Number of images {} and Labels  {} don't match".format(len(self.images), len(self.labels)))

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
        img = transform_functional.to_tensor(img)

        target = Image.open(self.labels[index]).convert('1')  # [0,1] Mask
        target = transform_functional.to_tensor(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data_set_mean_and_std(self):
        """
        Compute the data set mean and standard deviation
        Var[x] = E[X ^ 2] - E ^ 2[X]
        Ref: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
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

    data_set_dir = './data/edge_detection_data_set'

    plt.ion()
    batch_size = 16

    # Imagenet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # # Objective of Normalization is to set mean to 0 and standard deviation to 1. These values
    # # are from self.data_set_mean, self.data_set_std
    # normalize = transforms.Normalize(
    #     mean=[0.2587, 0.2587, 0.2587],
    #     std=[0.1074, 0.1074, 0.1074]
    # )

    # -------------------------------------------
    print("Setting up the Train Data Loaders")

    train_set = EdgeDataSet(data_dir=os.path.join(data_set_dir, 'train'), transform=normalize)

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    train_generator = training_data_loader.__iter__()
    train_imgs, train_labels = train_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))
    print("Batch mean = {}. std = {}".format(
        torch.mean(train_imgs, dim=[0, 2, 3]), torch.std(train_imgs, dim=[0, 2, 3])))

    # -------------------------------------------
    print("Setting up the Test Data Loader")
    test_set = EdgeDataSet(data_dir=os.path.join(data_set_dir, 'val'), transform=normalize)

    test_data_loader = DataLoader(
        dataset=test_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_generator = test_data_loader.__iter__()
    test_imgs, test_labels = test_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))

    for i_idx in np.arange(batch_size):

        image = train_imgs[i_idx, ].numpy()
        image = np.transpose(image, axes=(1, 2, 0))

        label = train_labels[i_idx, ].numpy()
        label = label.squeeze()

        f, ax_arr = plt.subplots(1, 2)
        ax_arr[0].imshow(image)
        ax_arr[0].set_title("Image")

        ax_arr[1].imshow(label)
        ax_arr[1].set_title("Label")

        import pdb
        pdb.set_trace()

    # -----------------------------------------------------------------------------------
    input("Press any key to exit")
