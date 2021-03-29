from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from models.vgg import VGG19
from models.mlp import MLP

from initializers.common import check_architecture_is_sequential
from initializers.pca import initialize_pca
from initializers.basic import \
        initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform

from cifar_dataloaders import create_CIFAR10_dataloaders
from librispeech_dataloaders import create_librispeech_dataloaders


def create_all_SPPs(train_loader, val_loader):
    raise Exception("Deprecated")

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_he(model_relu, train_loader)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "He ReLU")

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_pca(model_relu, train_loader)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "PCA ReLU")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32),  "PCA Tanh")

    model_relu = VGG19(num_classes = 10)
    vgg_initialize_pca(model_relu, train_loader, zca = True)
    signal_propagation_plot(model_relu, (5, 3, 32, 32), "ZCA ReLU")

    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    vgg_initialize_pca(model_tanh, train_loader, zca = True)
    signal_propagation_plot(model_tanh, (5, 3, 32, 32),  "ZCA Tanh")

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

def signal_propagation_plot(model, input_shape, name):
    inputs = torch.normal(mean = torch.zeros(input_shape))

    model_relu = VGG19(num_classes = 10)
    with SignalPropagationPlotter(model, name):
        model(inputs)

class SignalPropagationPlotter:
    def __init__(self, model, filename):
        check_architecture_is_sequential(model)
        self.model = model
        self.filename = filename


    def __enter__(self):
        self.stats, self.handles = self.register_hooks(self.model)


    def __exit__(self, exc_type, exc_value, traceback):
        for i in self.handles:
            i.remove()

        fig, axs = plt.subplots(2)
        fig.suptitle(f"{self.filename}: Avg. Channel Squared Mean / Avg. Channel Variance")
        axs[0].plot(self.stats[0])
        axs[1].plot(self.stats[1])
        plt.savefig(f"images/{self.filename}.png")


    def activation_hook(self, stats_dict, idx):
        def forward_hook(self1, layer_input, layer_output):
            average_channel_squared_mean_act = self.average_channel_squared_mean(layer_output)
            average_channel_variance_act = self.average_channel_variance(layer_output)

            stats_dict[0].append(average_channel_squared_mean_act)
            stats_dict[1].append(average_channel_variance_act)

        return forward_hook


    def average_channel_squared_mean(self, tensor):
        if len(tensor.shape) == 4:
            mean_per_channel = tensor.mean(0).mean(1).mean(1)
            squared_mean_per_channel = mean_per_channel ** 2
            return squared_mean_per_channel.mean()
        else:
            mean_per_neuron = tensor.mean(0)
            squared_mean_per_neuron = mean_per_neuron ** 2
            return squared_mean_per_neuron.mean()


    def average_channel_variance(self, tensor):
        if len(tensor.shape) == 4:
            channel_first = tensor.transpose(0, 1)
            num_channels = tensor.shape[0]
            for_var_calculation = channel_first.reshape(num_channels, -1)
            variance_per_channel = for_var_calculation.var(dim = 1)
            return variance_per_channel.mean()
        else:
            variance_per_neuron = tensor.var(0)
            return variance_per_neuron.mean()


    def register_hooks(self, model):
        stats_dict = ([], [])
        handles = []
        for idx, conv in enumerate(self.get_affine_layers(model)):
            hook = self.activation_hook(stats_dict, idx)
            handle = conv.register_forward_hook(hook)
            handles.append(handle)

        return stats_dict, handles


    def get_affine_layers(self, model):
        affine_types = (nn.Conv2d, nn.Linear)
        return [i for i in model.layers if isinstance(i, affine_types)]



if __name__ == "__main__":
    main()
