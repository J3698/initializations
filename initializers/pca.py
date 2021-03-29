import torch
import torch.nn as nn
import numpy as np
from initializers.common import *
from initializers.lsuv import iteratively_scale_and_rebias_conv_layer
from torch.utils.data import DataLoader
import tqdm
from math import ceil
from models.vgg import VGG19
from cifar_dataloaders import create_CIFAR10_dataloaders

AFFINE_TYPES = (nn.Linear, nn.Conv2d)

def verbose_test():
    train_loader, _ = create_CIFAR10_dataloaders()
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader, verbose = False, show_progress = True)


def initialize_pca(model: nn.Module, train_loader: DataLoader, zca = False,\
                       show_progress = False, verbose = False) -> None:
    model = model.cuda()
    model.train()

    print("Checking that model supports PCA")
    check_model_supports_pca(model)

    print("Getting batch of all inputs")
    last_layers_output = get_batch_of_all_inputs(train_loader, show_progress)

    layers = tqdm.tqdm(model.layers) if show_progress else model.layers

    for layer in layers:
        initialize_pca_if_conv2d(layer, last_layers_output, zca = zca, verbose = verbose)
        initialize_pca_if_linear(layer, last_layers_output, zca = zca, verbose = verbose)
        last_layers_output = put_all_batches_through_layer(layer, last_layers_output)
        if torch.isnan(last_layers_output.mean()):
            print("Error")
        if torch.isnan(last_layers_output.var()):
            print("Error")

        if verbose:
            with torch.no_grad():
                var = s_ntorch.var(last_layers_output)
                print(var)


def initialize_pca_if_conv2d(layer, data_orig: torch.Tensor,
                             zca = False, verbose = False) -> None:

    if not isinstance(layer, nn.Conv2d): return

    data = reshape_and_transpose_batches_for_conv_pca(data_orig)

    # PCA / ZCA
    # print(len(data))
    data = data[:2000]
    s, V = sorted_pcad_data(data)

    s[s < 1e-6] = 0
    s[s >= 1e-6] = 1 / torch.sqrt(s[s >= 1e-6] + 1e-3)
    S = torch.diag(s)

    weight = S @ V.T
    if zca:
        weight = V @ weight

    weight = weight.view(-1)[-layer.weight.shape.numel():]
    weight = weight.reshape(layer.weight.shape)


    with torch.no_grad():
        layer.weight[...] = weight
        layer.bias[...] = 0

    iteratively_scale_and_rebias_conv_layer(layer, data_orig, verbose = verbose)


def initialize_pca_if_linear(layer, data_orig: torch.Tensor,
                             zca = False, verbose = False) -> None:

    if not isinstance(layer, nn.Linear): return

    #print(data_orig.shape)

    data = data_orig.reshape(-1, data_orig.shape[-1])

    # PCA / ZCA
    #print(len(data))
    data = data[:2048]
    s, V = sorted_pcad_data(data)

    s[s < 1e-6] = 0
    s[s >= 1e-6] = 1 / torch.sqrt(s[s >= 1e-6] + 1e-3)
    S = torch.diag(s)

    weight = S @ V.T
    if zca:
        weight = V @ weight
    weight = weight.cuda()

    weight = weight.view(-1)[-layer.weight.shape.numel():]
    weight = weight.reshape(layer.weight.shape)


    with torch.no_grad():
        layer.weight[...] = weight
        layer.bias[...] = 0

    iteratively_scale_and_rebias_linear_layer(layer, data_orig, verbose = verbose)



def sorted_pcad_data(data):
    data = data.cpu()
    with torch.no_grad():
        data = data - data.mean(dim = 0, keepdims = True)
        Z = data @ data.T
        s, V = np.linalg.eigh(Z)
        sorted_indices = np.argsort(s)
        s = s[sorted_indices]
        V = V[:, sorted_indices]

        return torch.from_numpy(s), torch.from_numpy(V)


def reshape_and_transpose_batches_for_conv_pca(batches):
    data = batches.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)
    data = data.transpose(1, 2)
    data = data.reshape(-1, data.shape[-1])

    return data


def check_model_supports_pca(model: nn.Module) -> None:
    supported_layers = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten, nn.BatchNorm2d, nn.AvgPool2d)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)

