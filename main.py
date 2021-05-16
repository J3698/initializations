#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from dataloaders import create_MNIST_dataloaders,\
                        create_CIFAR10_dataloaders,\
                        create_librispeech_dataloaders

from models.mlp import MLP, MLPBN, SmallMLP
from models.vgg import VGG19

from util.signal_propagation_plots import SignalPropagationPlotter
from train import test_all_inits


def main():
    cuda = torch.cuda.is_available()
    NUM_WORKERS = os.cpu_count()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

    print("Creating data loaders")

    dataset = 'mnist'
    if dataset == "libri":
        train_loader, val_loader = create_librispeech_dataloaders(15, BATCH_SIZE)
        batch_size = 4096 if cuda else 1
        num_epochs = 2
        models = [MLP]

    elif dataset == "mnist":
        batch_size = 4096 if cuda else 512
        train_loader, val_loader = create_MNIST_dataloaders(True, batch_size)
        num_epochs = 2
        models = [SmallMLP]

    elif dataset == "cifar":
        train_loader, val_loader = create_CIFAR10_dataloaders(BATCH_SIZE)
        batch_size = 4096 if cuda else 1
        num_epochs = 2
        models = [VGG19]

    else:
        raise Exception("Dataset must be one of libri, mnist, cifar!")

    writer = SummaryWriter()
    test_all_inits(train_loader, val_loader, models, 
                   writer, spp = False, fcp = False, mds = False)


if __name__ == "__main__":
    main()


