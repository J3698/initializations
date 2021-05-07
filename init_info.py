import torch
import torch.nn as nn

from initializers.basic import *
from initializers.pca import *
from models.mlp import *
from models.vgg import *

init_types = {\
    #initialize_tanh_xavier_uniform: {\
    #    "include_nonlinearities": [nn.Tanh]
    #},
    #initialize_tanh_lecun_uniform: {\
    #    "include_nonlinearities": [nn.Tanh]
    #},
    #initialize_he: {\
    #    "include_nonlinearities": [nn.ReLU]
    #},
    initialize_orthogonal: {},
    #initialize_lsuv_pca: {},
    initialize_lsuv_zca: {},
    #initialize_pca: {},
    initialize_lsuv_kmeans: {},
    initialize_lsuv_random_samples: {},
    #initialize_random_samples: {},
    #initialize_zca: {},
    #initialize_kmeans: {},
}

model_types = [MLP, MLPBN, VGG19, VGG19BN]

nonlinearity_types = [nn.ReLU]#, nn.Tanh]
