import torch
import torch.nn as nn
from efficientnet import film_efficientnet, film_efficientnet_b0_84


def create_feature_extractor(args):
    if args.image_size == 84:
        feature_extractor = film_efficientnet_b0_84(args.pretrained_model_path)
    else:
        feature_extractor = film_efficientnet("efficientnet-b0")

    # freeze the parameters of feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


def create_film_adapter(feature_extractor, task_dim):
    adaptation_layer = FilmLayerGenerator
    adaptation_config = feature_extractor.get_adaptation_config()
    feature_adapter = FilmAdapter(
        layer=adaptation_layer,
        adaptation_config=adaptation_config,
        task_dim=task_dim
    )

    return feature_adapter


class BaseFilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(BaseFilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_generated_params = 0

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class FilmLayer(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim=None):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)

        self.gammas, self.gamma_regularizers = nn.ParameterList(), nn.ParameterList()
        self.betas, self.beta_regularizers = nn.ParameterList(), nn.ParameterList()

        for i in range(self.num_blocks):
            self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001), requires_grad=True))
            self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001), requires_grad=True))

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gammas[block] * self.gamma_regularizers[block] + torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.betas[block] * self.beta_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params


class FilmAdapter(nn.Module):
    def __init__(self, layer, adaptation_config, task_dim=None):
        super().__init__()
        self.num_maps = adaptation_config['num_maps_per_layer']
        self.num_blocks = adaptation_config['num_blocks_per_layer']
        self.task_dim = task_dim
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.num_generated_params = 0
        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    task_dim=self.task_dim
                )
            )
            self.num_generated_params += layers[-1].num_generated_params
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class DenseBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, in_size)
        self.layernorm = nn.LayerNorm(in_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.layernorm(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class FilmLayerGenerator(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        self.task_dim = task_dim

        self.gamma_generators, self.gamma_regularizers = nn.ModuleList(), nn.ParameterList()
        self.beta_generators, self.beta_regularizers = nn.ModuleList(), nn.ParameterList()

        for i in range(self.num_blocks):
            self.num_generated_params += 2 * num_maps[i]
            self.gamma_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                        requires_grad=True))

            self.beta_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                       requires_grad=True))

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseBlock(in_size, out_size)

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gamma_generators[block](x).squeeze() * self.gamma_regularizers[block] +
                         torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.beta_generators[block](x).squeeze() * self.beta_regularizers[block],
            }
            block_params.append(block_param_dict)
        return block_params


