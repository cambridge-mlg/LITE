import torch
import numpy as np
import torch.nn as nn
from config_networks import ConfigureNetworks
from mahalanonbis import MahalanobisPredictor
from set_encoder import mean_pooling


class FewShotClassifier(nn.Module):
    def __init__(self, args, logger, device):
        super(FewShotClassifier, self).__init__()
        self.args = args
        self.logger = logger
        self.device = device
        networks = ConfigureNetworks(args=self.args)
        self.set_encoder = networks.get_encoder()
        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.classifier = MahalanobisPredictor()
        self.means = None
        self.precisions = None
        self.reps_cache = None

    def forward(self, context_images, context_labels, target_images, meta_learning_state):
        self.build_task_representation(context_images)
        context_features, target_features = self.get_features(context_images, target_images, meta_learning_state)
        self.means, self.precisions = self.classifier.compute_class_means_and_precisions(context_features, context_labels)
        return self.classifier.predict(target_features, self.means, self.precisions)

    def configure_classifier(self, context_features, context_labels):
        self.means, self.precisions = self.classifier.compute_class_means_and_precisions(context_features, context_labels)

    def predict(self, target_images, meta_learning_state):
        target_features = self.get_target_features(target_images, meta_learning_state)
        return self.classifier.predict(target_features, self.means, self.precisions)

    def build_task_representation(self, context_images):
        self.task_representation = self.set_encoder(context_images)

    def build_task_representation_by_batch(self, context_images):
        reps = []
        num_images = context_images.size(0)
        num_batches = int(np.ceil(float(num_images) / float(self.args.batch_size)))
        for batch in range(num_batches):
            batch_start_index, batch_end_index = self._get_batch_indices(batch, num_images)
            reps.append(self.set_encoder.pre_pool(context_images[batch_start_index: batch_end_index]))
        self.task_representation = mean_pooling(torch.vstack(reps))

    def build_task_representation_with_split_batch(self, context_images, grad_indices, no_grad_indices):
        num_images = context_images.size(0)
        if self.reps_cache is None:  # cache the part with no gradients
            reps = []
            num_batches = int(np.ceil(float(num_images) / float(self.args.batch_size)))
            for batch in range(num_batches):
                batch_start_index, batch_end_index = self._get_batch_indices(batch, num_images)
                torch.set_grad_enabled(False)
                reps.append(self.set_encoder.pre_pool(context_images[batch_start_index: batch_end_index]))
            self.reps_cache = torch.vstack(reps).to(self.device)

        # now select some random images for that will have gradients and process those
        embeddings = []
        if len(grad_indices) > 0:
            torch.set_grad_enabled(True)
            embeddings.append(self.set_encoder.pre_pool(context_images[grad_indices]))

        # now add in the no_grad images
        embeddings.extend(self.reps_cache[no_grad_indices])

        # pool
        self.task_representation = mean_pooling(torch.vstack(embeddings))

    def get_context_features(self, context_images, meta_learning_state):
        feature_extractor_params = self.feature_adaptation_network(self.task_representation)
        self._set_batch_norm_mode(meta_learning_state)
        return self.feature_extractor(context_images, feature_extractor_params)

    def get_target_features(self, target_images, meta_learning_state):
        feature_extractor_params = self.feature_adaptation_network(self.task_representation)
        self._set_batch_norm_mode(meta_learning_state)
        return self.feature_extractor(target_images, feature_extractor_params)

    def get_features(self, context_images, target_images, meta_learning_state):
        feature_extractor_params = self.feature_adaptation_network(self.task_representation)
        self._set_batch_norm_mode(meta_learning_state)
        context_features = self.feature_extractor(context_images, feature_extractor_params)
        self._set_batch_norm_mode(meta_learning_state)
        target_features = self.feature_extractor(target_images, feature_extractor_params)
        return context_features, target_features

    def _set_batch_norm_mode(self, meta_learning_state):
        self.feature_extractor.eval()  # ignore context and state flag

    def _get_batch_indices(self, index, last_element):
        batch_start_index = index * self.args.batch_size
        batch_end_index = batch_start_index + self.args.batch_size
        if batch_end_index == (last_element - 1):  # avoid batch size of 1
            batch_end_index = last_element
        if batch_end_index > last_element:
            batch_end_index = last_element
        return batch_start_index, batch_end_index

    def count_parameters(self, model):
        model_param_count = sum(p.numel() for p in model.parameters())
        model_trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        feature_extractor_param_count = sum(p.numel() for p in model.feature_extractor.parameters())
        set_encoder_param_count = sum(p.numel() for p in model.set_encoder.parameters())
        feature_adaptation_param_count = sum(p.numel() for p in model.feature_adaptation_network.parameters())

        self.logger.print_and_log('Parameter Counts:')
        self.logger.print_and_log('Model: {}'.format(model_param_count))
        self.logger.print_and_log('Trainable: {}'.format(model_trainable_param_count))
        self.logger.print_and_log('Feature Extractor: {}'.format(feature_extractor_param_count))
        self.logger.print_and_log('Set Encoder: {}'.format(set_encoder_param_count))
        self.logger.print_and_log('Feature Extractor Adaptation Network: {}'.format(feature_adaptation_param_count))

    def clear_caches(self):
        self.reps_cache = None
