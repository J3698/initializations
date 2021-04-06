import torch
import torch.nn as nn
import numpy as np
from initializers.common import *
from initializers.lsuv import create_scaling_based_init
from torch.utils.data import DataLoader
import tqdm
from math import ceil
from models.vgg import VGG19
from sklearn.cluster import KMeans
from cifar_dataloaders import create_CIFAR10_dataloaders


AFFINE_TYPES = (nn.Linear, nn.Conv2d)


def check_model_supports_pca(model: nn.Module) -> None:
    supported_layers = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten, nn.BatchNorm2d, nn.AvgPool2d, nn.Module)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


def verbose_test():
    train_loader, _ = create_CIFAR10_dataloaders()
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader, verbose = False, show_progress = True)


def initialize_layer_pca(layer, last_layers_output, verbose = False) -> None:
    initialize_pca_if_conv2d(layer, last_layers_output, zca = False, verbose = verbose)
    initialize_pca_if_linear(layer, last_layers_output, zca = False, verbose = verbose)


def initialize_layer_zca(layer, last_layers_output, verbose = False) -> None:
    initialize_pca_if_conv2d(layer, last_layers_output, zca = True, verbose = verbose)
    initialize_pca_if_linear(layer, last_layers_output, zca = True, verbose = verbose)

def initialize_layer_kmeans(layer, last_layers_output, verbose = False) -> None:
    initialize_kmeans_if_conv2d(layer, last_layers_output, zca = True, verbose = verbose)
    initialize_kmeans_if_linear(layer, last_layers_output, zca = True, verbose = verbose)


def initialize_layer_data(layer, last_layers_output) -> None:
    initialize_data_if_conv(layer, last_layers_output)
    initialize_data_if_linear(layer, last_layers_output)


def initialize_data_if_conv(layer, last_layers_output) -> None:
    if not isinstance(layer, nn.Conv2d): return

    with torch.no_grad():
        layer.bias[...] = 0

        batches, batch_size, feats, w, h = last_layers_output.shape
        last_layers_output = last_layers_output.reshape(-1, feats, w, h)

        nums_per_batch_item = last_layers_output[0].shape.numel()
        nums_in_weight = layer.weight.shape.numel()
        data_necessary = nums_in_weight // nums_per_batch_item + 1
        indices = torch.randperm(len(last_layers_output))[:data_necessary]

        weight_data = last_layers_output[indices].reshape(-1)[:nums_in_weight]
        layer.weight[...] = weight_data.reshape(layer.weight.shape)


def initialize_data_if_linear(layer, last_layers_output) -> None:
    if not isinstance(layer, nn.Linear): return

    with torch.no_grad():
        layer.bias[...] = 0

        batches, batch_size, feats, w, h = last_layers_output.shape
        last_layers_output = last_layers_output.reshape(-1, feats, w, h)

        nums_per_batch_item = last_layers_output[0].shape.numel()
        nums_in_weight = layer.weight.shape.numel()
        data_necessary = nums_in_weight // nums_per_batch_item + 1
        indices = torch.randperm(len(last_layers_output))[:data_necessary]

        weight_data = last_layers_output[indices].reshape(-1)[:nums_in_weight]
        layer.weight[...] = weight_data.reshape(layer.weight.shape)


initialize_lsuv_pca = create_scaling_based_init(initialize_layer_pca, check_model_supports_pca)
initialize_lsuv_pca.__name__ = "initialize_lsuv_pca"
initialize_lsuv_zca = create_scaling_based_init(initialize_layer_zca, check_model_supports_pca)
initialize_lsuv_zca.__name__ = "initialize_lsuv_zca"
initialize_lsuv_kmeans = create_scaling_based_init(initialize_layer_kmeans, None)
initialize_lsuv_kmeans.__name__ = "initialize_lsuv_kmeans"

initialize_pca = create_layer_wise_init(initialize_layer_pca, check_model_supports_pca)
initialize_pca.__name__ = "initialize_pca"
initialize_zca = create_layer_wise_init(initialize_layer_zca, check_model_supports_pca)
initialize_zca.__name__ = "initialize_zca"
initialize_kmeans = create_layer_wise_init(initialize_layer_kmeans, None)
initialize_kmeans.__name__ = "initialize_kmeans"


def initialize_kmeans_if_conv2d(layer, last_layers_output, zca = True, verbose = False):
    if not isinstance(layer, nn.Conv2d): return

    data = batches_to_one_batch(last_layers_output)
    b, x, w, h = data.shape
    data = data.reshape(b, -1)
    necessary = layer.weight.shape.numel() // data.shape[1] + 1
    km = KMeans(necessary).fit(data.cpu().detach().numpy()).cluster_centers_

    weight = km.reshape(-1)[:layer.weight.shape.numel()]

    with torch.no_grad():
        layer.weight[...] = torch.from_numpy(weight.reshape(layer.weight.shape)).cuda()
        layer.bias[...] = 0


def initialize_kmeans_if_linear(layer, last_layers_output, zca = True, verbose = False):
    if not isinstance(layer, nn.Linear): return

    data = batches_to_one_batch(last_layers_output)
    b, f  = data.shape
    necessary = layer.weight.shape.numel() // data.shape[1] + 1
    km = KMeans(necessary).fit(data.cpu().detach().numpy()).cluster_centers_
    weight = km.reshape(-1)[:layer.weight.shape.numel()]

    with torch.no_grad():
        layer.weight[...] = weight.reshape(layer.weight.shape)
        layer.bias[...] = 0

initialize_lsuv_random_samples = create_scaling_based_init(initialize_layer_data, \
                                                           check_architecture_is_sequential)
initialize_lsuv_random_samples.__name__ = "initialize_lsuv_random_samples"
initialize_random_samples = create_layer_wise_init(initialize_layer_data, \
                                                   check_architecture_is_sequential)
initialize_random_samples.__name__ = "initialize_random_samples "


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


