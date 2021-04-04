from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from itertools import islice
from functools import partial


def create_layer_wise_init(layer_init_function, model_checker):
    kwargs = { "layer_init_function": layer_init_function, "model_checker": model_checker }
    initializer = partial(layer_wise_initialize, **kwargs)
    return initializer

def layer_wise_initialize(model: nn.Module, train_loader, layer_init_function,
                          show_progress = False, verbose = False, model_checker = None) -> None:

    model = model.cuda()
    model.train()

    if model_checker is None:
        check_architecture_is_sequential(model)
    else:
        model_checker(model)

    last_layers_output = get_batch_of_all_inputs(train_loader, show_progress)

    layers = tqdm.tqdm(enumerate(model.layers)) if show_progress else enumerate(model.layers)

    warn = True
    for i, layer in layers:
        layer_init_function(layer, last_layers_output)
        last_layers_output = put_all_batches_through_layer(layer, last_layers_output)

        if warn:
            warn = not warn_about_infs_and_nans(layer, last_layers_output, i)

        if verbose:
            with torch.no_grad():
                var = s_ntorch.var(last_layers_output)
                print(var)


def warn_about_infs_and_nans(layer, layer_output, i):
    for param in layer.parameters():
        if torch.any(torch.isnan(param)):
            print(f"Warning: nans in params layer {i}")
            return True
        if torch.any(torch.isinf(param)):
            print(f"Warning: infs in params layer {i}")
            return True

    if torch.any(torch.isnan(layer_output)):
        print(f"Warning: nans in output layer {i}")
        return True

    if torch.any(torch.isinf(layer_output)):
        print(f"Warning: infs in output layer {i}")
        return True

    return False



class ArchitectureNotSupportedException(Exception):
    def __init__(self, model):
        self.model_string = repr(model)

    def __str__(self):
        return self.model_string


class LayerNotSupportedException(Exception):
    def __init__(self, layer):
        self.str = str(layer)

    def __str__(self):
        return self.str


def check_architecture_is_sequential(model):
    assert model.layers is not None
    assert isinstance(model.layers, nn.Sequential)


def check_all_layers_supported(model: nn.Sequential,
                             supported_layers: Tuple[type, ...]):
    for layer in model:
        if not isinstance(layer, supported_layers):
            raise LayerNotSupportedException(layer)


def get_first_batch_inputs(train_loader):
    data, labels = next(iter(train_loader))
    return data


def get_batch_of_all_inputs(train_loader: DataLoader, show_progress = False) -> torch.Tensor:
    torch.multiprocessing.set_sharing_strategy('file_system')

    max_items = len(train_loader) // 20
    train_loader = islice(train_loader, 0, max_items)
    if show_progress:
        train_loader = tqdm.tqdm(train_loader, total = max_items)


    # print(f"item: {next(iter(train_loader))[0].shape}, len: {max_items}")
    loader = enumerate(train_loader)
    data = [x[None, ...] for i, (x, y) in loader]
    return torch.cat(data, dim =  0).cuda()


def calc_channel_means_and_vars(layer, batches):
    out = put_all_batches_through_layer(layer, batches)
    as_one_batch = batches_to_one_batch(out)

    spatial_flattened = flatten_spatial_dims(as_one_batch)

    channel_means = means_per_channel(spatial_flattened)
    channel_vars = mean_vars_per_channeel(spatial_flattened)

    return channel_means, channel_vars

def calc_neuron_means_and_vars(layer, batches):
    out = put_all_batches_through_layer(layer, batches)
    as_one_batch = batches_to_one_batch(out)
    neuron_means = tensor.mean(0)
    variance_per_neuron = tensor.var(0)

    assert neuron_means.shape == (layer.out_features,)
    assert variance_per_neuron.shape == (layer.out_features,)

    return neuron_means, variance_per_neuron



def calc_avg_squared_mean_and_avg_var(channel_means, channel_vars):
    avg_squared_mean = (channel_means ** 2).mean()
    avg_var = channel_vars.mean()

    return avg_squared_mean, avg_var

def means_per_channel(data):
    data = data.transpose(0, 1)
    data = data.reshape(data.shape[0], -1)

    return data.mean(dim = 1)


def mean_vars_per_channeel(data):
    var_per_channel_and_instance = torch.var(data, dim = 2)
    avg_var_per_channel = var_per_channel_and_instance.mean(dim = 0)
    return avg_var_per_channel


def batches_to_one_batch(batches):
    total_instances = batches.shape[0] * batches.shape[1]
    new_shape = (total_instances,) + batches.shape[2:]

    return batches.reshape(new_shape)


def flatten_spatial_dims(batch):
    batch_size, channels, width, height = batch.shape
    return batch.reshape(batch_size, channels, width * height)


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
