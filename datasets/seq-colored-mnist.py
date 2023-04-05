from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    CHMAP = {
        "red": (torch.tensor([1, 0, 0]), torch.tensor([255.0, 0, 0])),
        "green": (torch.tensor([0, 1, 0]), torch.tensor([0, 255.0, 0])),
        "blue": (torch.tensor([0, 0, 1]), torch.tensor([0, 0, 255.0])),
        "black": (0, 0)
    }

    def __init__(self, root, colors, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)
        self.colors = colors
        self.n_tasks = len(self.classes) // len(colors)
        self.train = train

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
        bg_color, fg_color = self.colors[target // self.n_tasks]

        img[img < 50] = 0.0
        img[img >= 50] = 255.0

        color_img[img != 0] *= MyMNIST.CHMAP[fg_color][0]
        color_img[img == 0] = MyMNIST.CHMAP[bg_color][1]

        img = Image.fromarray(color_img.numpy().astype(np.uint8))

        if self.train:
            original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        if self.train:
            return img, target, original_img
        else:
            return img, target

    def debugger(self, color_img: torch.Tensor):
        import matplotlib.pyplot as plt
        plt.imshow(color_img.numpy())
        plt.show()


# class MyMNIST(MNIST):
#     """
#     Overrides the MNIST dataset to change the getitem function.
#     """
#     # CHMAP = {
#     #     "red": 0,
#     #     "green": 1,
#     #     "blue": 2,
#     # }
#     CHMAP = {
#         "red": torch.Tensor([255., 0., 0.]),
#         "green": torch.Tensor([0., 255., 0.]),
#         "blue": torch.Tensor([0., 0., 255.]),
#     }
#
#     def __init__(self, root, colors, train=True, transform=None,
#                  target_transform=None, download=False) -> None:
#         self.not_aug_transform = transforms.ToTensor()
#         super(MyMNIST, self).__init__(root, train,
#                                       transform, target_transform, download)
#         self.colors = colors
#         self.n_tasks = len(self.classes) // len(colors)
#         self.train = train

    # def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
    #     """
    #     Gets the requested element from the dataset.
    #     :param index: index of the element to be returned
    #     :returns: tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     h, w = img.shape
    #     color_img = torch.zeros((h, w, 3))
    #
    #     fg_color = self.colors[target // self.n_tasks]
    #     color_img[:, :, MyMNIST.CHMAP[fg_color]] = img
    #     self.debugger(color_img)
    #
    #     img = Image.fromarray(color_img.numpy().astype(np.uint8))
    #
    #     if self.train:
    #         original_img = self.not_aug_transform(img.copy())
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     if self.train and hasattr(self, 'logits'):
    #         return img, target, original_img, self.logits[index]
    #
    #     if self.train:
    #         return img, target, original_img
    #     else:
    #         return img, target

    # def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
    #     """
    #     Gets the requested element from the dataset.
    #     :param index: index of the element to be returned
    #     :returns: tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     h, w = img.shape
    #     color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
    #
    #     bg_color = self.colors[target // self.n_tasks]
    #     color_img[img < 50] = MyMNIST.CHMAP[bg_color]
    #     #self.debugger(color_img)
    #
    #     img = Image.fromarray(color_img.numpy().astype(np.uint8))
    #
    #     if self.train:
    #         original_img = self.not_aug_transform(img.copy())
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     if self.train and hasattr(self, 'logits'):
    #         return img, target, original_img, self.logits[index]
    #
    #     if self.train:
    #         return img, target, original_img
    #     else:
    #         return img, target
    #
    # def debugger(self, color_img: torch.Tensor):
    #     import matplotlib.pyplot as plt
    #     plt.imshow(color_img.numpy())
    #     plt.show()


class ColorSequentialOODMNIST(ContinualDataset):

    NAME = 'col-seq-ood-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = None
    IID_COLORS = [("red", "black"), ("green", "black"), ("blue", "black"), ("black", "red"), ("black", "green")]
    #IID_COLORS = ["red", "green", "blue", "red", "green"]
    #IID_COLORS = ["green", "green", "blue", "red", "green"]

    def get_data_loaders(self, base_data_path, ood=[]):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_data_path + 'MNIST',
                                train=True, download=True, transform=transform, colors=ColorSequentialOODMNIST.IID_COLORS)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            if not ood:
                colors = ColorSequentialOODMNIST.IID_COLORS
            else:
                colors = [ColorSequentialOODMNIST.IID_COLORS[i] for i in ood]
            test_dataset = MyMNIST(base_data_path + 'MNIST',
                                   train=False, download=True, transform=transform,
                                   colors=colors)
                                   #colors=ColorSequentialOODMNIST.IID_COLORS)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size, base_data_path):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_data_path + 'MNIST',
                                train=True, download=True, transform=transform, colors=ColorSequentialOODMNIST.IID_COLORS)
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i -
            self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, ColorSequentialOODMNIST.N_TASKS
                        * ColorSequentialOODMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None