#  --------------------------------------------------------------------------------------
#   Pytorch Generator for the Fields 1993 Dataset
#  --------------------------------------------------------------------------------------
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import fields1993_stimuli


def get_filtered_gabor_sets(file_names, gabor_set_arr):
    use_set = []

    if gabor_set_arr:
        for set_idx in gabor_set_arr:
            use_set.extend([x for x in file_names if 'frag_{}/'.format(set_idx) in x])
    else:
        use_set = file_names

    return use_set


def get_filtered_c_len_files(file_names, c_len_arr):
    use_set = []

    if c_len_arr:
        for c_len in c_len_arr:
            use_set.extend([x for x in file_names if 'clen_{}_'.format(c_len) in x])
    else:
        use_set = file_names

    return use_set


def get_filtered_beta_files(file_names, beta_arr):
    use_set = []

    if beta_arr:
        for beta in beta_arr:
            use_set.extend([x for x in file_names if 'beta_{}_'.format(beta) in x])
    else:
        use_set = file_names

    return use_set


def get_filtered_alpha_files(file_names, alpha_arr):
    use_set = []

    if alpha_arr:
        for alpha in alpha_arr:
            use_set = [x for x in file_names if 'alpha_{}'.format(alpha) in x]
    else:
        use_set = file_names

    return use_set


def get_filtered_file_names(file_names, c_len_arr, beta_arr, alpha_arr, gabor_set_arr):
    """
    Filter out all file_names that contained the required c_len, beta and alpha values

    :param gabor_set_arr:
    :param file_names:
    :param c_len_arr:
    :param beta_arr:
    :param alpha_arr:
    :return:
    """

    use_set = get_filtered_gabor_sets(file_names, gabor_set_arr)
    use_set = get_filtered_c_len_files(use_set, c_len_arr)
    use_set = get_filtered_beta_files(use_set, beta_arr)
    use_set = get_filtered_alpha_files(use_set, alpha_arr)

    return use_set


class Fields1993(Dataset):
    def __init__(self, data_dir, bg_tile_size, augment=False, transform=None,
                 c_len_arr=None, beta_arr=None, alpha_arr=None, gabor_set_arr=None, subset_size=None):

        self.data_dir = data_dir
        self.bg_tile_size = bg_tile_size
        self.transform = transform
        self.augment = augment      # Todo: Add

        self.restrictions = {
            'c_len_arr': c_len_arr,
            'beta_att': beta_arr,
            'alpha_arr': alpha_arr,
            'gabor_set_arr': gabor_set_arr,
            'subset_size': subset_size
        }

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

        file_names = get_filtered_file_names(file_names, c_len_arr, beta_arr, alpha_arr, gabor_set_arr)
        # for idx, file_name in enumerate(file_names):
        #     print("{}: {}".format(idx, file_name))

        if not file_names:
            raise Exception("No Files in Data set!")

        if subset_size is not None:
            assert subset_size < len(file_names), 'subset size {} is greater than dataset size {}'.format(
                subset_size, len(file_names))

            use_idxs = np.random.choice(np.arange(len(file_names)), size=subset_size, replace=False)
            use_file_names = [file_names[idx] for idx in use_idxs]
            file_names = use_file_names

        # Sort the list of files
        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.labels = [os.path.join(label_dir, x + ".npy") for x in file_names]

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        sort_idxs = np.argsort(self.images)

        self.images = self.images[sort_idxs]
        self.labels = self.labels[sort_idxs]

        self.images = self.images.tolist()
        self.labels = self.labels.tolist()

        assert (len(self.images) == len(self.labels))

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

        target = np.load(self.labels[index])
        target = torch.from_numpy(np.array(target)).unsqueeze(0)

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

    plt.ion()
    batch_size = 16

    # # Imagenet
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    # Objective of Normalization is to set mean to 0 and standard deviation to 1. These values
    # are from self.data_set_mean, self.data_set_std
    normalize = transforms.Normalize(
        mean=[0.2587, 0.2587, 0.2587],
        std=[0.1074, 0.1074, 0.1074]
    )

    # -------------------------------------------
    print("Setting up the Train Data Loaders")
    train_set = Fields1993(data_dir="./data/curved_contours/train", bg_tile_size=(18, 18), transform=normalize)

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

    print("Setting up the Test Data Loader")
    test_set = Fields1993(data_dir='./data/curved_contours/test',  bg_tile_size=(18, 18))

    test_data_loader = DataLoader(
        dataset=train_set,
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
        label = np.squeeze(label, axis=0)

        fields1993_stimuli.plot_label_on_image(
            image,
            label,
            train_set.bg_tile_size
        )

    # -----------------------------------
    input("Press any key to exit")
