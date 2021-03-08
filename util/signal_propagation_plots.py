import torch
import torch.nn as nn
from initializers.common import check_architecture_is_sequential
from collections import defaultdict
import matplotlib.pyplot as plt
from models.vgg import VGG19

from initializers.pca import vgg_initialize_pca
from initializers.basic import \
        vgg_initialize_he, vgg_initialize_orthogonal, \
        vgg_initialize_tanh_lecun_uniform, vgg_initialize_tanh_xavier_uniform
from dataloaders import create_CIFAR10_dataloaders


def main():
    train_loader, val_loader = create_CIFAR10_dataloaders()

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_he(model_relu)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "He ReLU")

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_pca(model_relu, train_loader)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "PCA ReLU")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32),  "PCA Tanh")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_tanh_lecun_uniform(model_tanh)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32), "LeCun Uniform Tanh")

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_orthogonal(model_relu)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "Orthogonal")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_orthogonal(model_tanh)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32), "Orthogonal")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_tanh_xavier_uniform(model_tanh)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32), "Xavier Tanh")



def signal_propagation_plot(model, input_shape, filename):
    check_architecture_is_sequential(model)

    stats = register_hooks(model)
    inputs = torch.normal(mean = torch.zeros(input_shape))

    with torch.no_grad():
        model(inputs)

    fig, axs = plt.subplots(2)
    fig.suptitle(f"{filename}: Average Channel Squared Mean and Average Channel Variance")
    axs[0].plot(stats[0])
    axs[1].plot(stats[1])
    plt.savefig(f"images/{filename}.png")


def print_test(stats_dict, idx):
    def forward_hook(self, layer_input, layer_output):
        average_channel_squared_mean_act = average_channel_squared_mean(layer_output)
        average_channel_variance_act = average_channel_variance(layer_output)

        stats_dict[0].append(average_channel_squared_mean_act)
        stats_dict[1].append(average_channel_variance_act)

    return forward_hook


def average_channel_squared_mean(tensor):
    mean_per_channel = tensor.mean(0).mean(1).mean(1)
    squared_mean_per_channel = mean_per_channel ** 2
    return squared_mean_per_channel.mean()


def average_channel_variance(tensor):
    channel_first = tensor.transpose(0, 1)
    num_channels = tensor.shape[0]
    for_var_calculation = channel_first.reshape(num_channels, -1)
    variance_per_channel = for_var_calculation.var(dim = 1)
    return variance_per_channel.mean()


def register_hooks(model):
    stats_dict = ([], [])
    for idx, conv in enumerate(get_conv2d_layers(model)):
        conv.register_forward_hook(print_test(stats_dict, idx))

    return stats_dict


def get_conv2d_layers(model):
    return [i for i in model.layers if isinstance(i, nn.Conv2d)]


if __name__ == "__main__":
    main()
