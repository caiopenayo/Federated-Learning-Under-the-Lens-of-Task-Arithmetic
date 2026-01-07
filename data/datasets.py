# data/datasets.py

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def get_cifar100_transforms(img_size=160):
    """
    Retorna transforms de treino e teste compatíveis com ViT/DINO.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_transform, test_transform


def get_cifar100(train_transform, test_transform, val_ratio=0.1, root="./data", seed=42):
    """
    Retorna:
      train_subset, val_subset, test_dataset
    """
    full_train = datasets.CIFAR100(
        root=root, train=True, download=True, transform=train_transform
    )

    test = datasets.CIFAR100(
        root=root, train=False, download=True, transform=test_transform
    )

    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [n_train, n_val], generator=generator)

    return train, val, test
