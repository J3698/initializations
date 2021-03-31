from initializers.common import *
from functools import partial


def create_scaling_based_init(layer_init_function, model_checker):
    def layer_init_function(layer, last_layers_output):
        layer_init_function(layer, last_layers_output)
        iteratively_scale_and_rebias_layer(layer, last_layers_output)

    return create_layer_wise_init(layer_init_function, model_checker)


lsuv = create_scaling_based_init(lambda x, y: None, None)

"""
def scaling_based_initialize(model: nn.Module, dataloader, layer_init_function,
                             show_progress = False, verbose = False, model_checker = None) -> None:
    model = model.cuda()
    model.train()

    print("Checking that model is valid with this init")
    if model_checker is None:
        check_architecture_is_sequential(model)
    else:
        model_checker(model)

    print("Getting batch of all inputs")
    last_layers_output = get_batch_of_all_inputs(train_loader, show_progress)

    layers = tqdm.tqdm(model.layers) if show_progress else model.layers

    for layer in layers:
        layer_init_function(layer, last_layers_output)
        iteratively_scale_and_rebias_linear_layer(layer, last_layers_output, verbose = verbose)
        last_layers_output = put_all_batches_through_layer(layer, last_layers_output)

        if verbose:
            with torch.no_grad():
                var = s_ntorch.var(last_layers_output)
                print(var)
"""


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
