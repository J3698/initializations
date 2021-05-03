import torch
import torchvision
from torchvision import transforms
from six.moves import urllib


def main():
    create_CIFAR10_dataloaders()
    create_MNIST_dataloaders()


def create_CIFAR10_dataloaders(batch_size = 1):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                              shuffle=False, num_workers=2)

    print(f"CIFAR10 train set length {len(trainset)}, train loader length {len(train_loader)}")
    print(f"CIFAR10 test set length {len(testset)}, test loader length {len(test_loader)}")

    return train_loader, test_loader


def create_MNIST_dataloaders(flatten = False, batch_size = 1):
    fix_MNIST_download_issue()

    transforms_to_compose = [transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))]
    if flatten:
        transforms_to_compose.append(torch.flatten)

    transform = transforms.Compose(transforms_to_compose)

    trainset = torchvision.datasets.MNIST(root = './data', train = True,
                                            download = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                               shuffle = True, num_workers = 2)

    testset = torchvision.datasets.MNIST(root = './data', train = False,
                                           download = True, transform = transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                              shuffle = False, num_workers = 2)

    print(f"MNIST train set length {len(trainset)}, train loader length {len(train_loader)}")
    print(f"MNIST test set length {len(testset)}, test loader length {len(test_loader)}")

    return train_loader, test_loader


def fix_MNIST_download_issue():
    # https://github.com/pytorch/vision/issues/1938#issuecomment-594623431
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


if __name__ == "__main__":
    main()
