from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from initializers.common import check_architecture_is_sequential


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
        plt.savefig(f"{self.filename}.png")
        plt.close()


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


