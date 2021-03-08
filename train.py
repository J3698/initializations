import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from dataloaders import create_CIFAR10_dataloaders
from initializers.pca import vgg_initialize_pca
from initializers.basic import \
        vgg_initialize_he, vgg_initialize_orthogonal, \
        vgg_initialize_tanh_lecun_uniform, vgg_initialize_tanh_xavier_uniform
from util.signal_propagation_plots import signal_propagation_plot
from models.vgg import VGG19



def main():
    train_loader, val_loader = create_CIFAR10_dataloaders()
    model = VGG19(num_classes = 10)

    vgg_initialize_he(model)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_epoch(model, optimizer, criterion, train_loader)


def check_all_inits_work(model, train_loader):
    model_relu = VGG19(num_classes = 10)
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)

    vgg_initialize_he(model_relu)
    vgg_initialize_pca(model_relu, train_loader)
    vgg_initialize_pca(model_tanh, train_loader)
    vgg_initialize_tanh_lecun_uniform(model_tanh)
    vgg_initialize_orthogonal(model_relu)
    vgg_initialize_orthogonal(model_tanh)
    vgg_initialize_tanh_xavier_uniform(model_tanh)
vgg_initialize_tanh_lecun_uniform


def train_epoch(model, optimizer, criterion, train_loader):
    correct, total = 0, 0
    total_loss = 0
    for i, (x, y) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += (y == y_pred.argmax(-1)).sum()
        total += len(y)

def validate(model, criterion, val_loader):
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in train_loader:
            y_pred = model(x)
            total_loss += criterion(y_pred, y).item()
            correct += (y == y_pred.argmax(-1)).sum()
            total += len(y)

    return total / correct




if __name__ == "__main__":
    main()


