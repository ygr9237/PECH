from re import A
from PIL import Image, ImageFile
import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageFilter
from loguru import logger


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Onehot(object):
    def __call__(self, sample, num_class=31):
        target_onehot = torch.zeros(num_class)
        target_onehot[sample] = 1

        return target_onehot


def train_transform():
    """
    Training images transform.
    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def train_aug_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def query_transform():
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def load_data(source_list, target_list, batch_size, num_workers, task='cross'):
    office.init(source_list, target_list, task)
    query_dataset = office('query', query_transform(), target_transform=Onehot())
    train_s_dataset = office('train_s', train_transform(), target_transform=Onehot())
    retrieval_dataset = office('retrieval', query_transform(), target_transform=Onehot())
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_s_dataloader = DataLoader(
        train_s_dataset,
        batch_size=batch_size,
        shuffle=True,  # 对数据进行洗牌
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, train_s_dataloader, retrieval_dataloader


class office(Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.aug = train_aug_transform()
        self.mode = mode
        self.num_class = 31

        if mode == 'train_s':
            self.data = office.TRAIN_S_DATA
            self.targets = office.TRAIN_S_TARGETS

        elif mode == 'query':
            self.data = office.QUERY_DATA
            self.targets = office.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = office.RETRIEVAL_DATA
            self.targets = office.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        img_aug = self.aug(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_aug, self.target_transform(self.targets[index]), index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        one_hot = torch.zeros((self.targets.shape[0], self.num_class))
        for i in range(self.targets.shape[0]):
            one_hot[i, :] = self.target_transform(self.targets[i])
        return one_hot

    @staticmethod
    def init(source_list, target_list, task):
        source_data = []
        source_label = []
        target_data = []
        target_label = []
        with open(source_list, 'r') as f:
            for line in f:
                source_data.append(line.split()[0].replace( \
                    '/data', '/data'))
                source_label.append(int(line.split()[1]))

        with open(target_list, 'r') as f:
            for line in f:
                target_data.append(line.split()[0].replace( \
                    '/data', '/data'))
                target_label.append(int(line.split()[1]))

        source_data = np.array(source_data)
        source_label = np.array(source_label)
        target_data = np.array(target_data)
        target_label = np.array(target_label)

        if task == 'cross':

            perm_index = np.random.permutation(target_data.shape[0])
            query_index = perm_index[:int(0.1 * target_data.shape[0])]
            database_index = perm_index[int(0.1 * target_data.shape[0]):]

            office.QUERY_DATA = target_data[query_index]
            office.QUERY_TARGETS = target_label[query_index]

            office.TRAIN_S_DATA = source_data
            office.TRAIN_S_TARGETS = source_label

            office.RETRIEVAL_DATA = source_data
            office.RETRIEVAL_TARGETS = source_label

            logger.info('Query Num: {}'.format(office.QUERY_DATA.shape[0]))
            logger.info('Retrieval Num: {}'.format(office.RETRIEVAL_DATA.shape[0]))
            logger.info('Source Train Num: {}'.format(office.TRAIN_S_DATA.shape[0]))

        else:
            perm_index = np.random.permutation(target_data.shape[0])
            query_index = perm_index[:int(0.1 * target_data.shape[0])]
            database_index = perm_index[int(0.1 * target_data.shape[0]):]

            office.QUERY_DATA = target_data[query_index]
            office.QUERY_TARGETS = target_label[query_index]

            office.TRAIN_S_DATA = source_data
            office.TRAIN_S_TARGETS = source_label

            office.RETRIEVAL_DATA = target_data[database_index]
            office.RETRIEVAL_TARGETS = target_label[database_index]

            logger.info('Query Num: {}'.format(office.QUERY_DATA.shape[0]))
            logger.info('Retrieval Num: {}'.format(office.RETRIEVAL_DATA.shape[0]))
            logger.info('Source Train Num: {}'.format(office.TRAIN_S_DATA.shape[0]))
