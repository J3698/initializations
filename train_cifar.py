#!/usr/bin/env python3

import torch
from math import inf
import os
from contextlib import ExitStack
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import tqdm
from cifar_dataloaders import create_CIFAR10_dataloaders
from initializers.pca import *
from initializers.basic import \
        initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform
from util.signal_propagation_plots import signal_propagation_plot, SignalPropagationPlotter
from models.vgg import VGG19, VGG19BN
from tests import check_all_inits_work

cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

BATCH_SIZE = 64 if cuda else 1
NUM_EPOCHS = 45

def main():
    print("Creating data loaders")
    train_loader, val_loader = create_CIFAR10_dataloaders(BATCH_SIZE)

    check_all_inits_work(train_loader, val_loader)
    raise 1

    writer = SummaryWriter()

    model_relu = VGG19BN(num_classes = 10)
    initialize_random_samples(model_relu, train_loader)
    test_init("Random (ReLU) (BN)", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    initialize_lsuv_random_samples(model_relu, train_loader)
    test_init("Random (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19BN(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_random_samples(model_tanh, train_loader)
    test_init("Random (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)

    #model_relu = VGG19(num_classes = 10)
    #initialize_lsuv_kmeans(model_relu, train_loader)
    #test_init("KMeans (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_lsuv_random_samples(model_tanh, train_loader)
    test_init("Random (Tanh)", model_tanh, train_loader, val_loader, writer)


    model_relu = VGG19BN(num_classes = 10)
    initialize_kmeans(model_relu, train_loader)
    test_init("KMeans (ReLU) (BN)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10)
    initialize_lsuv_kmeans(model_tanh, train_loader)
    test_init("KMeans (Tanh)", model_tanh, train_loader, val_loader, writer)

    model_tanh = VGG19BN(num_classes = 10)
    initialize_kmeans(model_tanh, train_loader)
    test_init("KMeans (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)


    """
    model_tanh = VGG19BN(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    test_init("PCA (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)

    initialize_zca(model_tanh, train_loader)
    test_init("PCA (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)

    initialize_tanh_lecun_uniform(model_tanh)
    test_init("PCA (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)

    initialize_orthogonal(model_tanh)
    test_init("PCA (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)

    initialize_tanh_xavier_uniform(model_tanh)
    test_init("PCA (Tanh) (BN)", model_tanh, train_loader, val_loader, writer)
    """




    """
    model_relu = VGG19(num_classes = 10)
    initialize_pca(model_relu, train_loader)
    test_init("PCA (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    test_init("PCA Tanh", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    initialize_pca(model_relu, train_loader)
    test_init("ZCA (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    test_init("ZCA Tanh", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    initialize_he(model_relu)
    test_init("He ReLU", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_tanh_lecun_uniform(model_tanh)
    test_init("LeCun Uniform Tanh", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    initialize_orthogonal(model_relu)
    test_init("Orthogonal (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_orthogonal(model_tanh)
    test_init("Orthogonal (Tanh)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_tanh_xavier_uniform(model_tanh)
    test_init("Xavier Tanh", model_relu, train_loader, val_loader, writer)
    """


def test_init(init_name, model, train_loader, val_loader, writer):
    print(f"Testing intit: {init_name}")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.95)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_epoch(model, optimizer, criterion, train_loader, epoch, init_name)
        val_loss, val_accuracy = validate(model, criterion, val_loader)

        scheduler.step()

        writer.add_scalar(f'{init_name}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/train', train_accuracy, epoch)
        writer.add_scalar(f'{init_name}/Loss/validate', val_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/validate', val_accuracy, epoch)
        print(f"stats: {train_loss:.2f}, {100 * train_accuracy:.2f}%, {val_loss:.2f}, {100 * val_accuracy:.2f}%")
# misc-reading-group@cs.cmu.edu
#



def train_epoch(model, optimizer, criterion, train_loader, epoch, init_name):
    correct, total = 0.0, 0.0
    total_loss = 0.0

    model.train()

    spp = SignalPropagationPlotter(model, f"{init_name}-{epoch}")
    print(epoch)
    for i, (x, y) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader),\
                               dynamic_ncols = True):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        with spp if i == 0 else ExitStack():
            y_pred = model(x)

        loss = criterion(y_pred, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += (y == y_pred.argmax(-1)).sum().item()
        total += len(y)

    return total_loss / total, correct / total

def validate(model, criterion, val_loader):
    correct, total = 0.0, 0.0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(val_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)
            total_loss += criterion(y_pred, y).item()
            correct += (y == y_pred.argmax(-1)).sum().item()
            total += len(y)

    return total_loss / total, correct / total




if __name__ == "__main__":
    main()


