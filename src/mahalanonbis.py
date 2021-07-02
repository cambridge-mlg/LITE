import torch

"""
The code in this file is substantially based on the code for "Improved Few-Shot Visual Classification"
by Peyman Bateni, Raghav Goyal, Vaden Masrani1, Frank Wood, and Leonid Sigal
that can be found here: https://github.com/peymanbateni/simple-cnaps
"""

class MahalanobisPredictor:
    def __init__(self):
        return

    def predict(self, target_features, class_means, class_precision_matrices):
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(0)
        number_of_targets = target_features.size(0)

        """
        Calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
        repeated_class_means = class_means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        return logits

    def compute_class_means_and_precisions(self, features, labels):
        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(features)
        for c in torch.unique(labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(features, 0, self._extract_class_indices(labels, c))
            # mean pooling examples to form class means
            means.append(self._mean_pooling(class_features).squeeze())
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            precisions.append(torch.inverse(
                (lambda_k_tau * self._estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate)\
                + torch.eye(class_features.size(1), class_features.size(1)).cuda(0)))

        means = (torch.stack(means))
        precisions = (torch.stack(precisions))

        return means, precisions

    @staticmethod
    def _estimate_cov(examples, rowvar=False, inplace=False):
        """
        SCM: function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)
