from features import create_feature_extractor, create_film_adapter
from set_encoder import SetEncoder

""" Creates the set encoder, feature extractor, feature adaptation networks.
"""


class ConfigureNetworks:
    def __init__(self, args):
        self.encoder = SetEncoder()
        z_g_dim = self.encoder.pre_pooling_fn.output_size
        self.feature_extractor = create_feature_extractor(args)
        self.film_adapter = create_film_adapter(self.feature_extractor, z_g_dim)

    def get_encoder(self):
        return self.encoder

    def get_feature_adaptation(self):
        return self.film_adapter

    def get_feature_extractor(self):
        return self.feature_extractor
