from initializers.common import *


def iteratively_scale_and_rebias_linear_layer(layer, batches, max_iters = 5,
                                      verbose = False):
    """
        Iteratively scale weights and set bias so output var is 1, mean is 0.
        Stops when var / mean are close enough to 1 / 0 respectively, or max
        iterations have been reached
    """

    neuron_means, neuron_vars = calc_neuron_means_and_vars(layer, batches)

    max_iters_left = max_iters

    while max_iters_left > 0 and (avg_squared_mean > 1e-10 or abs(avg_var - 1) > 1e-10):
        scale_and_rebias_linear_layer(layer, neuron_means, neuron_vars)

        neuron_means, neuron_vars = calc_neuron_means_and_vars(layer, batches)

        max_iters_left -= 1

        if verbose:
            print(f"avg squared mean: {avg_squared_mean}")
            print(f"avg var: {avg_var}")

    print(neuron_means, neuron_vars)
    return 


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
        print("ran out of iters")
    else:
        print("converged")


def scale_and_rebias_layer(layer, channel_means, channel_vars):
    """
        Scale weights and set bias so output var is 1, mean is 0
    """

    with torch.no_grad():
        layer.weight /= channel_vars[:, None, None, None] ** 0.5
        layer.bias -= channel_means


def scale_and_rebias_linear_layer(layer, neuron_means, neuron_vars):
    """
        Scale weights and set bias so output var is 1, mean is 0
    """

    with torch.no_grad():
        layer.weight /= channel_vars[:, None, None, None] ** 0.5
        layer.bias -= channel_means
