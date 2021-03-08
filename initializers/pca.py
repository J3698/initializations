import torch
import torch.nn as nn
from initializers.common import *
from torch.utils.data import DataLoader
import tqdm
from math import ceil

def vgg_initialize_pca(model: nn.Module, train_loader: DataLoader, show_progress = False) -> None:
    check_model_supports_pca(model)
    last_layers_output = get_batch_of_all_inputs(train_loader, show_progress = False)
    layers = tqdm.tqdm(model.layers) if show_progress else model.layers
    for layer in model.layers:
        initialize_pca_if_conv2d(layer, last_layers_output)
        last_layers_output = put_all_batches_through_layer(layer, last_layers_output)


def initialize_pca_if_conv2d(layer: nn.Conv2d, data: torch.Tensor) -> None:
    if not isinstance(layer, nn.Conv2d): return

    data = data.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)
    data = data.transpose(1, 2)
    data = data.reshape(-1, data.shape[-1])

    # PCA
    data = data[:2000]
    # print("prepped")
    Z = data @ data.T
    # print("mat muled")
    s, V = torch.linalg.eigh(Z)
    # print("eighed")
    sorted_indices = torch.argsort(s)
    s = s[sorted_indices]
    V = V[:, sorted_indices]
    s[s < 1e-6] = 0
    s[s >= 1e-6] = 1 / torch.sqrt(s[s >= 1e-6] + 1e-3)
    S = torch.diag(s)
    weight = S @ V.T
    weight = weight.view(-1)[0: layer.weight.shape.numel()]
    weight = weight.reshape(layer.weight.shape)
    with torch.no_grad():
        layer.weight[...] = weight
        layer.bias[...] = 0


#   conv2d_initialize_pca(layer, last_layers_outputs)
def put_all_batches_through_layer(layer, batches):
    num_batches, batch_size = batches.shape[0:2]
    non_batch_dims = batches.shape[2:]

    output = []
    for batch in batches:
        output.append(layer(batch)[None, ...])
        del batch

    catted = torch.cat(output, 0)

    del output
    del batches

    return catted


def calculate_required_num_batches(model, dataloader):
    last_layers_output = get_first_batch_inputs(dataloader)
    for layer in model.layers:
        last_layers_output = layer(last_layers_output)


def check_model_supports_pca(model: nn.Module) -> None:
    supported_layers = (nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


