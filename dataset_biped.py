# ---------------------------------------------------------------------------------------
# Pytorch Data set/loaders for the Barcelona Images for perceptual Edge detection (BIPED)
# Dataset.
# Ref: https://github.com/xavysp/MBIPED
# Ref: Data: https://drive.google.com/file/d/1l9cUbNK7CgpUsWYInce-djJQp-FyY5DO/view
# Ref: Paper: https://arxiv.org/pdf/1909.01955.pdf
# ---------------------------------------------------------------------------------------
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as transform_functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class BipedDataSet(Dataset):
    def __init__(self, data_dir, dataset_type='train', transform=None, subset_size=None,
                 resize_size=None):

        self.transform = transform
        self.resize_size = resize_size

        if resize_size is not None:
            self.resize = transforms.Resize(resize_size)
        else:
            self.resize = None

        if not os.path.exists(data_dir):
            raise Exception("Cannot find data dir {}".format(data_dir))

        self.data_dir = data_dir

        valid_dataset_types = ['train', 'test']
        if dataset_type not in valid_dataset_types:
            raise Exception("Invalid data set {}. Must be one of {}".format(
                dataset_type, valid_dataset_types))

        img_dir = os.path.join(data_dir, 'imgs/' + dataset_type)

        # Get a list of all files
        list_of_files = []

        for dir_path, dir_name, filenames in os.walk(img_dir):
            # print("For dir_path {}, dir_name {}:".format(dir_path, dir_name))
            # print("filenames {}".format(filenames))

            if filenames:
                for file in filenames:
                    list_of_files.append(os.path.join(dir_path, file))

        list_of_files = sorted(list_of_files)
        # print("Number of Data points:{} ".format(len(list_of_files)))

        self.images = list_of_files
        self.labels = \
            [file.replace('imgs', 'edge_maps').replace('.jpg', '.png') for file in self.images]

        if subset_size is not None:
            assert subset_size < len(
                self.images), 'subset size {} is greater than dataset size {}'.format(
                subset_size, len(self.images))

            use_idxs = np.random.choice(np.arange(len(self.images)), size=subset_size,
                                        replace=False)
            self.images = [self.images[idx] for idx in use_idxs]
            self.labels = [self.labels[idx] for idx in use_idxs]

        if len(self.images) != len(self.labels):
            raise Exception(
                "Number of images {} and Labels  {} don't match".format(
                    len(self.images), len(self.labels)))

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
        target = Image.open(self.labels[index]).convert('L')  # Greyscale

        if self.resize is not None:
            img = self.resize(img)
            target = self.resize(target)  # uses interpolation

        img = transform_functional.to_tensor(img)
        target = transform_functional.to_tensor(target)
        target[target > 0.1] = 1  # necessary for smooth contours after interpolation

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data_set_mean_and_std(self):
        """
        Compute the data set mean and standard deviation
        Var[x] = E[X ^ 2] - E ^ 2[X]

        Ref:
        https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
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
    base_dir = '/home/salman/workspace/pytorch/MBIPED/dataset/BIPED/edges'

    # Images are of different sizes and cannot be batched together.
    # Resizing the image appears to cause the edge labels to becoming jagged and
    # discontinuous
    train_batch_size = 32
    test_batch_size = 1

    # Immutable
    import matplotlib.pyplot as plt
    plt.ion()

    # Imagenet normalization
    ch_mean = [0.485, 0.456, 0.406]
    ch_std = [0.229, 0.224, 0.225]
    pre_process_transforms = transforms.Normalize(mean=ch_mean, std=ch_std)

    # -----------------------------------------------------------------------------------
    # Training Loader
    # -----------------------------------------------------------------------------------
    print("Setting up the Train Data Loaders {}".format('*'*30))
    start_time = datetime.now()

    train_set = BipedDataSet(
        data_dir=base_dir,
        dataset_type='train',
        transform=pre_process_transforms,
        subset_size=100,
        resize_size=(256, 256)
    )

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("Setting up the train dataloader took {}".format(datetime.now() - start_time))

    train_generator = training_data_loader.__iter__()
    train_imgs, train_labels = train_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(train_imgs.shape, train_labels.shape))
    print("Batch mean = {}. std = {}".format(
        torch.mean(train_imgs, dim=[0, 2, 3]), torch.std(train_imgs, dim=[0, 2, 3])))

    # Visualize Some
    for img_idx in range(train_imgs.shape[0]):
        image = train_imgs[img_idx, ].numpy()
        image = np.transpose(image, axes=(1, 2, 0))
        label = train_labels[img_idx, ].numpy()
        label = np.squeeze(label)

        f, ax_arr = plt.subplots(1, 2)
        ax_arr[0].imshow(image)
        ax_arr[0].set_title("Image")

        ax_arr[1].imshow(label)
        ax_arr[1].set_title("Label")

    # # Debug - Show images using the generator
    # img_idx = 0
    # for iteration, (image, label) in enumerate(training_data_loader, 1):
    #     image = image[img_idx, ].numpy()
    #     image = np.transpose(image, axes=(1, 2, 0))
    #
    #     label = label[img_idx, ].numpy()
    #     label = np.squeeze(label)
    #
    #     f, ax_arr = plt.subplots(1, 2)
    #     ax_arr[0].imshow(image)
    #     ax_arr[0].set_title("Image, size {}".format(image.shape))
    #
    #     ax_arr[1].imshow(label)
    #     ax_arr[1].set_title("Label, size {}".format(label.shape))
    #
    #     import pdb
    #     pdb.set_trace()
    #
    #     plt.close(f)

    # -----------------------------------------------------------------------------------
    # Test Loader
    # -----------------------------------------------------------------------------------
    print("Setting up the Train Data Loaders, {}".format("*"*30))
    start_time = datetime.now()

    test_set = BipedDataSet(
        data_dir=base_dir,
        dataset_type='test',
        transform=pre_process_transforms,
    )

    test_data_loader = DataLoader(
        dataset=test_set,
        num_workers=4,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True
    )

    print("setting up the test data loader took {}".format(datetime.now() - start_time))

    test_generator = test_data_loader.__iter__()
    test_imgs, test_labels = test_generator.__next__()
    print("Images shape {}. Labels.shape {} ".format(test_imgs.shape, test_labels.shape))
    print("Batch mean = {}. std = {}".format(
        torch.mean(test_imgs, dim=[0, 2, 3]), torch.std(test_imgs, dim=[0, 2, 3])))

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    import pdb
    pdb.set_trace()
