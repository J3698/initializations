from initializers.common import *
import torch.nn as nn
import math
import tqdm


def initialize_tanh_xavier_uniform(model, loader, show_progress = False):
    check_model_supports_tanh_basic(model)

    layers = model.layers if not show_progress else tqdm.tqdm(model.layers)
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def initialize_tanh_lecun_uniform(model, loader, show_progress = False):
    check_model_supports_tanh_basic(model)

    layers = model.layers if not show_progress else tqdm.tqdm(model.layers)
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            lecun_uniform_init_weight(layer.weight, loader)
            torch.nn.init.zeros_(layer.bias)


# modified from torch source
def lecun_uniform_init_weight(weight, loader, show_progress = False):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = math.sqrt(2.0 / float(fan_in))
    a = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(weight, -a, a)


def check_model_supports_tanh_basic(model: nn.Module) -> None:
    supported_layers = (nn.Linear, nn.Conv2d, nn.Tanh, nn.Flatten, nn.Module)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


def initialize_he(model, loader, show_progress = False):
    check_model_supports_relu_he(model)

    layers = model.layers if not show_progress else tqdm.tqdm(model.layers)
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity = 'relu')
            torch.nn.init.zeros_(layer.bias)


def check_model_supports_relu_he(model: nn.Module) -> None:
    supported_layers = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Flatten, nn.Module)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


def initialize_orthogonal(model, loader, show_progress = False):
    check_model_supports_orthogonal(model)

    layers = model.layers if not show_progress else tqdm.tqdm(model.layers)
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def check_model_supports_orthogonal(model: nn.Module) -> None:
    supported_layers = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten, nn.Module)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


