import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import tqdm
from dataloaders import create_CIFAR10_dataloaders
from initializers.pca import vgg_initialize_pca
from initializers.basic import \
        vgg_initialize_he, vgg_initialize_orthogonal, \
        vgg_initialize_tanh_lecun_uniform, vgg_initialize_tanh_xavier_uniform
from util.signal_propagation_plots import signal_propagation_plot, create_all_SPPs
from models.vgg import VGG19


cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

BATCH_SIZE = 64 if cuda else 1
NUM_EPOCHS = 50

def main():
    train_loader, val_loader = create_CIFAR10_dataloaders(BATCH_SIZE)
    print("Created data loaders")

    # check_all_inits_work(train_loader, val_loader)
    print("Ran each init as a test")

    # create_all_SPPs(train_loader, val_loader)
    print("Created SPPs")

    writer = SummaryWriter()

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_he(model_relu)
    test_init("He ReLU", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_pca(model_relu, train_loader)
    test_init("PCA (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader)
    test_init("PCA Tanh", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_pca(model_relu, train_loader)
    test_init("ZCA (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader)
    test_init("ZCA Tanh", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_tanh_lecun_uniform(model_tanh)
    test_init("LeCun Uniform Tanh", model_relu, train_loader, val_loader, writer)

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_orthogonal(model_relu)
    test_init("Orthogonal (ReLU)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_orthogonal(model_tanh)
    test_init("Orthogonal (Tanh)", model_relu, train_loader, val_loader, writer)

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_tanh_xavier_uniform(model_tanh)
    test_init("Xavier Tanh", model_relu, train_loader, val_loader, writer)


def test_init(init_name, model, train_loader, val_loader, writer):
    print(f"Testing intit: {init_name}")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.95)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_accuracy = validate(model, criterion, val_loader)

        scheduler.step()

        writer.add_scalar(f'{init_name}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/train', train_accuracy, epoch)
        writer.add_scalar(f'{init_name}/Loss/validate', train_loss, epoch)
        writer.add_scalar(f'{init_name}/Accuracy/validate', train_accuracy, epoch)
        print(f"stats: {train_loss:.2f}, {100 * train_accuracy:.2f}%, {train_loss:.2f}, {100 * train_accuracy:.2f}%")
# misc-reading-group@cs.cmu.edu
#

def check_all_inits_work(train_loader, val_loader):
    model_relu = VGG19(num_classes = 10)
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)

    vgg_initialize_he(model_relu)
    vgg_initialize_pca(model_relu, train_loader)
    vgg_initialize_pca(model_tanh, train_loader)
    vgg_initialize_pca(model_relu, train_loader, zca = True)
    vgg_initialize_pca(model_tanh, train_loader, zca = True)
    vgg_initialize_tanh_lecun_uniform(model_tanh)
    vgg_initialize_orthogonal(model_relu)
    vgg_initialize_orthogonal(model_tanh)
    vgg_initialize_tanh_xavier_uniform(model_tanh)


def train_epoch(model, optimizer, criterion, train_loader):
    correct, total = 0.0, 0.0
    total_loss = 0.0

    model.train()
    for i, (x, y) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
 
        optimizer.zero_grad()
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


