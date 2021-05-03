import torch
import torch.nn as nn
import numpy as np
from initializers.common import *
from initializers.lsuv import create_scaling_based_init
from torch.utils.data import DataLoader
import tqdm
from math import ceil
from models.vgg import VGG19
from sklearn.cluster import MiniBatchKMeans
from cifar_dataloaders import create_CIFAR10_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def initialize_layer_kmeans(layer, last_layers_output, verbose = False, hook = None) -> None:
    initialize_kmeans_if_conv2d(layer, last_layers_output, verbose = verbose, hook = hook)
    initialize_kmeans_if_linear(layer, last_layers_output, verbose = verbose, hook = hook)


def initialize_layer_data(layer, last_layers_output) -> None:
    initialize_data_if_conv(layer, last_layers_output)
    initialize_data_if_linear(layer, last_layers_output)


def initialize_data_if_linear(layer, last_layers_output) -> None:
    if not isinstance(layer, nn.Linear): return
    assert len(last_layers_output.shape) == 3, "linear data should have shape (batches, batch_size, feats)"
    batches, batch_size, feats = last_layers_output.shape
    assert feats == layer.in_features, "data and layer don't match"

    with torch.no_grad():
        layer.bias[...] = 0

        last_layers_output = last_layers_output.reshape(-1, feats)
        assert last_layers_output.shape == (batches * batch_size, feats)

        total_inputs = last_layers_output.shape[0]
        inputs_to_take = torch.randperm(total_inputs)[:layer.out_features]
        new_weight = last_layers_output[inputs_to_take]
        assert new_weight.shape == layer.weight.shape

        layer.weight[...] = new_weight


def initialize_kmeans_if_linear(layer, last_layers_output, verbose = False, hook = None):
    if not isinstance(layer, nn.Linear): return
    assert len(last_layers_output.shape) == 3, "linear data should have shape (batches, batch_size, feats)"
    batches, batch_size, feats = last_layers_output.shape
    assert feats == layer.in_features, "data and layer don't match"

    data = batches_to_one_batch(last_layers_output)
    assert data.shape == (batches * batch_size, feats)
    necessary_centers = layer.out_features

    data_prepared = data.cpu().detach().numpy()[:max(necessary_centers * 10, len(data))]
    kmeans = MiniBatchKMeans(necessary_centers, n_init = 2, init_size = 2 * necessary_centers)
    new_weight = kmeans.fit(data_prepared).cluster_centers_
    assert new_weight.shape == layer.weight.shape

    with torch.no_grad():
        layer.weight[...] = torch.from_numpy(new_weight).to(DEVICE)
        layer.bias[...] = 0

    if hook is not None:
        hook(layer, last_layers_output)


def initialize_pca_if_linear(layer, last_layers_output: torch.Tensor, zca = False, verbose = False, hook = None) -> None:
    if not isinstance(layer, nn.Linear): return
    assert len(last_layers_output.shape) == 3, "linear data should have shape (batches, batch_size, feats)"
    batches, batch_size, feats = last_layers_output.shape
    assert feats == layer.in_features, "data and layer don't match"

    with torch.no_grad():
        data = last_layers_output.reshape(-1, feats)
        s, V = sorted_pcad_data(data, data.shape[1] * 10)
        assert V.shape == (feats, feats)

        if len(V) >= layer.out_features:
            weight = V[-layer.out_features:]
            assert weight.shape == layer.weight.shape, (weight.shape, layer.weight.shape)
        else:
            more_needed = (layer.out_features - len(V),)
            to_add = torch.randint(len(V), size = more_needed)
            weight = torch.cat((V[to_add], V), dim = 0)
            assert weight.shape == layer.weight.shape, (weight.shape, layer.weight.shape)

        weight = weight.to(DEVICE)
        layer.weight[...] = weight
        layer.bias[...] = 0

    if hook is not None:
        hook(layer, last_layers_output)


def initialize_data_if_conv(layer, last_layers_output) -> None:
    if not isinstance(layer, nn.Conv2d): return
    assert len(last_layers_output.shape) == 5, "conv data should have shape (batches, batch_size, channels, w, h)"
    batches, batch_size, channels, h, w = last_layers_output.shape
    out_channels, in_channels, kw, kh = layer.weight.shape
    assert channels == in_channels, "data and layer don't match"

    with torch.no_grad():
        last_layers_output = last_layers_output.reshape(batches * batch_size, channels, h, w)

        if h - layer.kernel_size[0] + 1 == 0:
            import pdb
            pdb.set_trace()

        rs = torch.randint(h - layer.kernel_size[0] + 1, (layer.out_channels,))
        cs = torch.randint(w - layer.kernel_size[1] + 1, (layer.out_channels,))
        batch_item = torch.randint(batches * batch_size, (layer.out_channels,))

        for b, r, c, o in zip(batch_item, rs, cs, range(layer.out_channels)):
            weight = last_layers_output[b, :, r: r + kh, c: c + kw]
            weight = weight.to(DEVICE)
            assert weight.shape == layer.weight[o].shape, (weight.shape, layer.weight[o].shape)
            layer.weight[o, ...] = weight

        layer.bias[...] = 0


def initialize_kmeans_if_conv2d(layer, last_layers_output, verbose = False, hook = None):
    if not isinstance(layer, nn.Conv2d): return
    assert len(last_layers_output.shape) == 5, "conv data should have shape (batches, batch_size, channels, w, h)"
    batches, batch_size, channels, h, w = last_layers_output.shape
    out_channels, in_channels, kh, kw = layer.weight.shape
    assert channels == in_channels, "data and layer don't match"


    with torch.no_grad():
        last_layers_output = last_layers_output.reshape(batches * batch_size, channels, h, w)

        if h - layer.kernel_size[0] + 1 == 0:
            import pdb
            pdb.set_trace()

        oversample = 5
        rs = torch.randint(h - layer.kernel_size[0] + 1, (oversample * layer.out_channels,))
        cs = torch.randint(w - layer.kernel_size[1] + 1, (oversample * layer.out_channels,))
        batch_item = torch.randint(batches * batch_size, (oversample * layer.out_channels,))

        data = []
        for b, r, c, o in zip(batch_item, rs, cs, range(oversample * layer.out_channels)):
            datum = last_layers_output[b, :, r: r + kh, c: c + kw]
            assert datum.shape == layer.weight[0].shape, (weight.shape, layer.weight[0].shape)
            data.append(datum.reshape(-1))

        data = torch.stack(data, dim = 0)
        assert data.shape == (oversample * layer.out_channels, channels * kh * kw)
        kmeans = MiniBatchKMeans(layer.out_channels, n_init = 2, init_size = 2 * layer.out_channels)
        centers = kmeans.fit(data.cpu().detach().numpy()).cluster_centers_
        weight = centers.reshape((layer.out_channels, layer.in_channels, kh, kw))

        layer.weight[...] = torch.from_numpy(weight)
        layer.bias[...] = 0




def initialize_pca_if_conv2d(layer, last_layers_output: torch.Tensor,
                             zca = False, verbose = False) -> None:
    if not isinstance(layer, nn.Conv2d): return
    assert len(last_layers_output.shape) == 5, "conv data should have shape (batches, batch_size, channels, w, h)"
    batches, batch_size, channels, h, w = last_layers_output.shape
    out_channels, in_channels, kh, kw = layer.weight.shape
    assert channels == in_channels, "data and layer don't match"

    with torch.no_grad():
        last_layers_output = last_layers_output.reshape(batches * batch_size, channels, h, w)

        if h - layer.kernel_size[0] + 1 == 0:
            import pdb
            pdb.set_trace()

        oversample = 5
        rs = torch.randint(h - layer.kernel_size[0] + 1, (oversample * layer.out_channels,))
        cs = torch.randint(w - layer.kernel_size[1] + 1, (oversample * layer.out_channels,))
        batch_item = torch.randint(batches * batch_size, (oversample * layer.out_channels,))

        data = []
        for b, r, c, o in zip(batch_item, rs, cs, range(oversample * layer.out_channels)):
            datum = last_layers_output[b, :, r: r + kh, c: c + kw]
            assert datum.shape == layer.weight[0].shape, (weight.shape, layer.weight[0].shape)
            data.append(datum.reshape(-1))

        data = torch.stack(data, dim = 0)
        assert data.shape == (oversample * layer.out_channels, channels * kh * kw)

        s, V = sorted_pcad_data(data)
        assert V.shape == (channels * kh * kw, channels * kh * kw), (V.shape, (channels * kh * kw, channels * kh * kw))

        if len(V) >= layer.out_channels:
            weight = V[-layer.out_channels:].reshape(layer.out_channels, layer.in_channels, kh, kw)
            assert weight.shape == layer.weight.shape, (weight.shape, layer.weight.shape)
        else:
            more_needed = (layer.out_channels - len(V),)
            to_add = torch.randint(len(V), size = more_needed)
            weight = torch.cat((V[to_add], V), dim = 0).reshape(layer.out_channels, layer.in_channels, kh, kw)
            assert weight.shape == layer.weight.shape, (weight.shape, layer.weight.shape)

        layer.weight[...] = weight
        layer.bias[...] = 0


# pca
initialize_lsuv_pca = create_scaling_based_init(initialize_layer_pca, check_model_supports_pca)
initialize_lsuv_pca.__name__ = "initialize_lsuv_pca"
initialize_pca = create_layer_wise_init(initialize_layer_pca, check_model_supports_pca)
initialize_pca.__name__ = "initialize_pca"

# zca
initialize_lsuv_zca = create_scaling_based_init(initialize_layer_zca, check_model_supports_pca)
initialize_lsuv_zca.__name__ = "initialize_lsuv_zca"
initialize_zca = create_layer_wise_init(initialize_layer_zca, check_model_supports_pca)
initialize_zca.__name__ = "initialize_zca"

# kmeans
initialize_lsuv_kmeans = create_scaling_based_init(initialize_layer_kmeans, None)
initialize_lsuv_kmeans.__name__ = "initialize_lsuv_kmeans"
initialize_kmeans = create_layer_wise_init(initialize_layer_kmeans, None)
initialize_kmeans.__name__ = "initialize_kmeans"

# random samples
initialize_lsuv_random_samples = create_scaling_based_init(initialize_layer_data, check_architecture_is_sequential)
initialize_lsuv_random_samples.__name__ = "initialize_lsuv_random_samples"
initialize_random_samples = create_layer_wise_init(initialize_layer_data, check_architecture_is_sequential)
initialize_random_samples.__name__ = "initialize_random_samples "


def sorted_pcad_data(data, min_data = None):
    with torch.no_grad():
        data = data - data.mean(dim = 0, keepdims = True)
        if min_data != None:
            indices = torch.randperm(len(data))[:min_data]
            data = data[indices]

        Z = data.T @ data

        ret = torch.eig(Z, eigenvectors = True)
        s, V = ret.eigenvalues[:, 0], ret.eigenvectors
        sorted_indices = torch.argsort(s)
        s = s[sorted_indices]
        V = V[:, sorted_indices]

        return s, V


def reshape_and_transpose_batches_for_conv_pca(batches):
    data = batches.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)
    data = data.transpose(1, 2)
    data = data.reshape(-1, data.shape[-1])

    return data


