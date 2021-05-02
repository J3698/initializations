#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms

import os
from contextlib import ExitStack
from torch.utils.tensorboard import SummaryWriter
import tqdm
from librispeech_dataloaders import create_librispeech_dataloaders
from initializers.pca import initialize_pca
from initializers.basic import initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform
from models.mlp import MLP, MLPBN
from tests import check_all_inits_work
from util.signal_propagation_plots import signal_propagation_plot, SignalPropagationPlotter
import init_info
from util.high_dim_visualize import visualize_weights_mlp
import traceback


cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

BATCH_SIZE = 4096 if cuda else 1
NUM_EPOCHS = 5

def main():
    print("Creating data loaders")
    train_loader, val_loader = create_librispeech_dataloaders(15, BATCH_SIZE)

    print("Generating MLP SPPs for each init")
    models = [MLP]
    # check_all_inits_work(train_loader, val_loader, models)

    writer = SummaryWriter()
    test_all_inits(train_loader, val_loader, models, writer)


def test_all_inits(train_loader, val_loader, models, writer):
    for init, info in init_info.init_types.items():
        for nonlinearity in init_info.nonlinearity_types:
            if "include_nonlinearities" in info and\
               nonlinearity not in info["include_nonlinearities"]:
                continue

            for model_type in models:
                test_name = f"{init.__name__}-{model_type.__name__}-" + \
                            f"{nonlinearity.__name__}"
                print(test_name)

                try:
                    model = model_type(nonlinearity = nonlinearity)
                    init(model, train_loader, show_progress = True)
                    test_init(test_name, model, train_loader, val_loader, writer, test_name)
                except Exception:
                    print(traceback.format_exc())
                    print("failed:", test_name)




def test_init(init_name, model, train_loader, val_loader, writer, test_name):
    print(f"Testing init: {init_name}")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.95)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        visualize_weights_mlp(model, train_loader, "./MDS", str(epoch), test_name)
        train_loss, train_accuracy = train_epoch(model, optimizer, criterion, train_loader, epoch, init_name)
        val_loss, val_accuracy = validate(model, criterion, val_loader)

        scheduler.step()

        writer.add_scalar(f'{init_name}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/train', train_accuracy, epoch)
        writer.add_scalar(f'{init_name}/Loss/validate', val_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/validate', val_accuracy, epoch)
        print(f"stats: {train_loss:.2f}, {100 * train_accuracy:.2f}%, {val_loss:.2f}, {100 * val_accuracy:.2f}%")


def train_epoch(model, optimizer, criterion, train_loader, epoch, init_name):
    correct, total = 0.0, 0.0
    total_loss = 0.0

    model.train()

    spp = SignalPropagationPlotter(model, f"{init_name}-{epoch}")
    for i, (x, y) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
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

