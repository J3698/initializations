import torch
import torch.nn as nn
from initializers.common import *
from initializers.lsuv import iteratively_scale_and_rebias_layer
from torch.utils.data import DataLoader
import tqdm
from math import ceil
from models.vgg import VGG19
from dataloaders import create_CIFAR10_dataloaders

def verbose_test():
    train_loader, _ = create_CIFAR10_dataloaders()
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader, verbose = False, show_progress = True)


def vgg_initialize_pca(model: nn.Module, train_loader: DataLoader, zca = False,\
                       show_progress = False, verbose = False) -> None:

    check_model_supports_pca(model)
    last_layers_output = get_batch_of_all_inputs(train_loader, show_progress = False)

    layers = tqdm.tqdm(model.layers) if show_progress else model.layers

    for layer in layers:
        initialize_pca_if_conv2d(layer, last_layers_output, zca = zca, verbose = verbose)
        last_layers_output = put_all_batches_through_layer(layer, last_layers_output)

        if verbose:
            with torch.no_grad():
                var = torch.var(last_layers_output)
                print(var)


def initialize_pca_if_conv2d(layer: nn.Conv2d, data_orig: torch.Tensor,
                             zca = False, verbose = False) -> None:

    if not isinstance(layer, nn.Conv2d): return

    data = reshape_and_transpose_batches_for_pca(data_orig)

    # PCA / ZCA
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

    iteratively_scale_and_rebias_layer(layer, data_orig, verbose = verbose)



def sorted_pcad_data(data):
    data = data - data.mean(dim = 0, keepdims = True)
    Z = data @ data.T
    s, V = torch.linalg.eigh(Z)
    sorted_indices = torch.argsort(s)
    s = s[sorted_indices]
    V = V[:, sorted_indices]

    return s, V


def reshape_and_transpose_batches_for_pca(batches):
    data = batches.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)
    data = data.transpose(1, 2)
    data = data.reshape(-1, data.shape[-1])

    return data


def check_model_supports_pca(model: nn.Module) -> None:
    supported_layers = (nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


