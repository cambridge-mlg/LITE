import os
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
import sys
import math


class MetaLearningState(Enum):
    META_TRAIN = 0
    META_TEST = 1


class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logger, accuracy_dict):
        logger.print_and_log("")  # add a blank line
        logger.print_and_log("Validation Accuracies:")
        for dataset in self.datasets:
            logger.print_and_log("{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                  accuracy_dict[dataset]["confidence"]))
        logger.print_and_log("")  # add a blank line

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict


class LogFiles:
    def __init__(self, checkpoint_dir, resume, test_mode):
        self._checkpoint_dir = checkpoint_dir
        if not self._verify_checkpoint_dir(resume, test_mode):
            sys.exit()
        if not test_mode and not resume:
            os.makedirs(self.checkpoint_dir)
        self._best_validation_model_path = os.path.join(checkpoint_dir, 'best_validation.pt')
        self._fully_trained_model_path = os.path.join(checkpoint_dir, 'fully_trained.pt')

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def best_validation_model_path(self):
        return self._best_validation_model_path

    @property
    def fully_trained_model_path(self):
        return self._fully_trained_model_path

    def _verify_checkpoint_dir(self, resume, test_mode):
        checkpoint_dir_is_ok = True
        if resume:  # verify that the checkpoint directory and file exists
            if not os.path.exists(self.checkpoint_dir):
                print("Can't resume from checkpoint. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            if not os.path.isfile(checkpoint_file):
                print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
                checkpoint_dir_is_ok = False

        elif test_mode:
            if not os.path.exists(self.checkpoint_dir):
                print("Can't test. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

        else:
            if os.path.exists(self.checkpoint_dir):
                print("Checkpoint directory ({}) already exits.".format(self.checkpoint_dir), flush=True)
                print("If starting a new training run, specify a directory that does not already exist.", flush=True)
                print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
                checkpoint_dir_is_ok = False

        return checkpoint_dir_is_ok


class Logger:
    def __init__(self, checkpoint_dir, log_file_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_file_path = os.path.join(checkpoint_dir, log_file_name)
        self.file = None
        if os.path.isfile(log_file_path):
            self.file = open(log_file_path, "a", buffering=1)
        else:
            self.file = open(log_file_path, "w", buffering=1)

    def __del__(self):
        self.file.close()

    def log(self, message):
        self.file.write(message + '\n')

    def print_and_log(self, message):
        print(message, flush=True)
        self.log(message)


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, torch.argmax(logits, dim=-1)).float())


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]
