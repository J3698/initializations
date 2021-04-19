from initializers.common import *
from functools import partial


def create_scaling_based_init(layer_init_function, model_checker):
    def new_layer_init_function(layer, last_layers_output, **kwargs):
        layer_init_function(layer, last_layers_output, **kwargs)
        iteratively_scale_and_rebias_layer(layer, last_layers_output)

    return create_layer_wise_init(new_layer_init_function, model_checker)


lsuv = create_scaling_based_init(lambda x, y: None, None)


def iteratively_scale_and_rebias_layer(layer, batches, max_iters = 5, verbose = False):
    if isinstance(layer, nn.Conv2d):
        iteratively_scale_and_rebias_conv_layer(layer, batches, max_iters, verbose)
    elif isinstance(layer, nn.Linear):
        iteratively_scale_and_rebias_linear_layer(layer, batches, max_iters, verbose)


def iteratively_scale_and_rebias_linear_layer(layer, batches, max_iters = 5,
                                      verbose = False):
    """
        Iteratively scale weights and set bias so output var is 1, mean is 0.
        Stops when var / mean are close enough to 1 / 0 respectively, or max
        iterations have been reached
    """

    neuron_means, neuron_vars = calc_neuron_means_and_vars(layer, batches)
    avg_squared_mean = (neuron_means ** 2).mean()
    avg_var = neuron_vars.mean()

    max_iters_left = max_iters

    while max_iters_left > 0 and (avg_squared_mean > 1e-10 or abs(avg_var - 1) > 1e-10):
        scale_and_rebias_linear_layer(layer, neuron_means, neuron_vars)

        neuron_means, neuron_vars = calc_neuron_means_and_vars(layer, batches)
        avg_squared_mean = (neuron_means ** 2).mean()
        avg_var = neuron_vars.mean()

        max_iters_left -= 1

        if verbose:
            print(f"avg squared mean: {avg_squared_mean}")
            print(f"avg var: {avg_var}")


def iteratively_scale_and_rebias_conv_layer(layer, batches, max_iters = 5,
                                      verbose = False):
    """
        Iteratively scale weights and set bias so output var is 1, mean is 0.
        Stops when var / mean are close enough to 1 / 0 respectively, or max
        iterations have been reached
    """

    channel_means, channel_vars = calc_channel_means_and_vars(layer, batches)
    avg_squared_mean, avg_var = calc_avg_squared_mean_and_avg_var(channel_means, channel_vars)

    max_iters_left = max_iters

    while max_iters_left > 0 and (avg_squared_mean > 1e-10 or abs(avg_var - 1) > 1e-10):
        scale_and_rebias_layer(layer, channel_means, channel_vars)

        channel_means, channel_vars = calc_channel_means_and_vars(layer, batches)
        avg_squared_mean, avg_var = calc_avg_squared_mean_and_avg_var(channel_means, channel_vars)

        max_iters_left -= 1

        if verbose:
            print(f"avg squared mean: {avg_squared_mean}")
            print(f"avg var: {avg_var}")

    if max_iters_left <= 0:
        print(f"Convergence failed, mean {avg_squared_mean:.2f} var {avg_var:.2f}")


def scale_and_rebias_layer(layer, channel_means, channel_vars):
    """
        Scale weights and set bias so output var is 1, mean is 0
    """

    with torch.no_grad():
        old_weight = layer.weight.clone()
        old_bias = layer.bias.clone()


        channel_vars[channel_vars == 0] = 1

        layer.weight /= channel_vars[:, None, None, None] ** 0.5
        layer.bias -= channel_means

        if torch.any(torch.isnan(layer.weight)) or torch.any(torch.isnan(layer.weight)) or \
           torch.any(torch.isinf(layer.weight)) or torch.any(torch.isinf(layer.weight)):
            import pdb
            pdb.set_trace()


def scale_and_rebias_linear_layer(layer, neuron_means, neuron_vars):
    """
        Scale weights and set bias so output var is 1, mean is 0
    """

    with torch.no_grad():
        neuron_vars[neuron_vars == 0] = 1
        layer.weight /= neuron_vars[:, None] ** 0.5
        layer.bias -= neuron_means
