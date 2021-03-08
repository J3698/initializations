from initializers.common import *
import torch.nn as nn
import math




def vgg_initialize_tanh_xavier_uniform(model):
    check_model_supports_tanh_basic(model)

    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def vgg_initialize_tanh_lecun_uniform(model):
    check_model_supports_tanh_basic(model)

    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            lecun_uniform_init_weight(layer.weight)
            torch.nn.init.zeros_(layer.bias)


# modified from torch source
def lecun_uniform_init_weight(weight):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    std = math.sqrt(2.0 / float(fan_in))
    a = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(weight, -a, a)


def check_model_supports_tanh_basic(model: nn.Module) -> None:
    supported_layers = (nn.Conv2d, nn.Tanh, nn.Flatten)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


def vgg_initialize_he(model):
    check_model_supports_relu_he(model)

    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity = 'relu')
            torch.nn.init.zeros_(layer.bias)


def check_model_supports_relu_he(model: nn.Module) -> None:
    supported_layers = (nn.Conv2d, nn.ReLU, nn.Flatten)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)


def vgg_initialize_orthogonal(model):
    check_model_supports_orthogonal(model)

    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def check_model_supports_orthogonal(model: nn.Module) -> None:
    supported_layers = (nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten)
    check_architecture_is_sequential(model)
    check_all_layers_supported(model.layers, supported_layers)

