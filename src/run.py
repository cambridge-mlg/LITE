import torch
import numpy as np
import argparse
import os
from utils import Logger, LogFiles, ValidationAccuracies, cross_entropy_loss, compute_accuracy, MetaLearningState,\
    shuffle
from model import FewShotClassifier
from dataset import get_dataset_reader
from tf_dataset_reader import TfDatasetReader
from image_folder_reader import ImageFolderReader

NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
PRINT_FREQUENCY = 1000


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.log_files = LogFiles(self.args.checkpoint_dir, self.args.resume_from_checkpoint,
                                  (self.args.mode == 'test') or (self.args.mode == 'test_vtab'))
        self.logger = Logger(self.args.checkpoint_dir, "log.txt")
        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.log_files.checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        if self.args.mode == "train" or self.args.mode == "test" or self.args.mode == 'train_test':
            self.dataset = get_dataset_reader(
                args=self.args,
                train_set=self.train_set,
                validation_set=self.validation_set,
                test_set=self.test_set)

        if self.args.train_method == 'lite':
            self.train_fn = self.train_lite
        else:
            self.train_fn = self.train_task

        self.use_batches = False if self.args.train_method == 'no_lite' else True

        self.loss = cross_entropy_loss
        self.accuracy_fn = compute_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.validation_accuracies = ValidationAccuracies(self.validation_set)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        self.feature_cache = None

    def init_model(self):
        model = FewShotClassifier(args=self.args, logger=self.logger, device=self.device).to(self.device)
        model.count_parameters(model)

        # set encoder is always in train mode (it only sees context data).
        # Feature extractor gets switched in model.
        model.train()
        return model

    def init_data(self):
        train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'mnist']
        validation_set = ['omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'mscoco']
        test_set = self.args.test_datasets

        return train_set, validation_set, test_set

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        # operational parameters
        parser.add_argument("--mode", choices=["train", "test", "train_test", "test_vtab"], default="train_test",
                            help="Whether to run meta-training only, meta-testing only,"
                                 "both meta-training and meta-testing, or testing on vtab.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")

        # data parameters
        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi", "traffic_sign",
                                     "mscoco"])
        parser.add_argument("--data_path", default="../datasets", help="Path to Meta-Dataset records.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--download_path_for_sun397_dataset", default=None,
                            help="Path to download the sun397 dataset.")

        # training parameters
        parser.add_argument("--train_method", choices=["lite", "small_task", "no_lite"], default="lite",
                            help="Whether to use lite, small tasks, or not lite.")
        parser.add_argument("--pretrained_model_path", default="../models/efficientnet-b0_84.pt",
                            help="Path to dataset records.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_step", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--training_iterations", "-i", type=int, default=15000,
                            help="Number of meta-training iterations.")
        parser.add_argument("--max_way_train", type=int, default=50, help="Maximum way of meta-train task.")
        parser.add_argument("--max_support_train", type=int, default=500,
                            help="Maximum support set size of meta-train task.")
        parser.add_argument("--image_size", type=int, default=224, help="Image height and width.")
        parser.add_argument("--batch_size", type=int, default=50, help="Size of batch.")

        # testing parameters
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--val_freq", type=int, default=5000, help="Number of iterations between validations.")

        args = parser.parse_args()

        return args

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations
            for iteration in range(self.start_iteration, total_iterations):
                task_dict = self.dataset.get_train_task()
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

                if self.use_batches:
                    self.model.clear_caches()
                    self.feature_cache = None
                    target_set_size = len(target_labels)
                    num_batches = self._get_number_of_batches(target_set_size)
                    for batch in range(num_batches):
                        batch_start_index, batch_end_index = self._get_batch_indices(batch, target_set_size)
                        batch_loss, batch_accuracy = self.train_fn(
                            context_images,
                            target_images[batch_start_index : batch_end_index],
                            context_labels,
                            target_labels[batch_start_index : batch_end_index]
                        )
                        train_accuracies.append(batch_accuracy)
                        losses.append(batch_loss)
                else:
                    task_loss, task_accuracy = self.train_fn(context_images, target_images, context_labels,
                                                                  target_labels)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_step == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iteration + 1) % PRINT_FREQUENCY == 0:
                    # print training stats
                    self.save_checkpoint(iteration + 1)
                    torch.save(self.model.state_dict(), os.path.join(self.log_files.checkpoint_dir,
                                                                     "model_{}.pt".format(iteration + 1)))
                    self.logger.print_and_log('Task [{}/{}], Train Loss: {:.7f},'
                                              'Train Accuracy: {:.7f}, Learning Rate: {:.7f}'
                                              .format(iteration + 1, total_iterations,
                                                      torch.Tensor(losses).mean().item(),
                                                      torch.Tensor(train_accuracies).mean().item(),
                                                      self.optimizer.param_groups[0]['lr']))
                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                    # validate
                    accuracy_dict = self.validate()
                    self.validation_accuracies.print(self.logger, accuracy_dict)
                    # save the model if validation is the best so far
                    if self.validation_accuracies.is_better(accuracy_dict):
                        self.validation_accuracies.replace(accuracy_dict)
                        torch.save(self.model.state_dict(), self.log_files.best_validation_model_path)
                        self.logger.print_and_log('Best validation model was updated.')
                        self.logger.print_and_log('')

            # save the final model
            torch.save(self.model.state_dict(), self.log_files.fully_trained_model_path)

        if self.args.mode == 'train_test':
            self.test(self.log_files.fully_trained_model_path)
            self.test(self.log_files.best_validation_model_path)

        if self.args.mode == 'test':
            self.test(self.args.test_model_path)

        if self.args.mode == 'test_vtab':
            self._test_transfer_learning(self.args.test_model_path)

    def train_task(self, context_images, target_images, context_labels, target_labels):
        target_logits = self.model(context_images, context_labels, target_images, MetaLearningState.META_TRAIN)
        task_loss = self.loss(target_logits, target_labels) / self.args.tasks_per_step
        regularization_term = (self.model.feature_adaptation_network.regularization_term())
        regularizer_scaling = 0.001
        task_loss += regularizer_scaling * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def train_lite(self, context_images, target_images, context_labels, target_labels):
        # We'll split the context set into two: the first part will be of size batch_size and we'll use gradients
        # for that. The second part will be everything else and we'll use no gradients for that, so we only need to
        # compute that once per task.
        indices = np.random.permutation(context_images.size(0))
        grad_indices = indices[0: self.args.batch_size]
        no_grad_indices = indices[self.args.batch_size:]

        self.model.build_task_representation_with_split_batch(context_images, grad_indices, no_grad_indices)
        context_features = self._compute_features_with_split_batch(context_images, grad_indices, no_grad_indices,
                                                                   MetaLearningState.META_TRAIN)
        self.model.configure_classifier(context_features, context_labels[indices])

        # now the target set
        torch.set_grad_enabled(True)
        batch_logits = self.model.predict(target_images, MetaLearningState.META_TRAIN)

        # compute the loss
        batch_loss = self.loss(batch_logits, target_labels) / self.args.tasks_per_step
        regularization_term = (self.model.feature_adaptation_network.regularization_term())
        regularizer_scaling = 0.001
        batch_loss += regularizer_scaling * regularization_term

        # compute accuracy
        batch_accuracy = self.accuracy_fn(batch_logits, target_labels)

        batch_loss.backward(retain_graph=False)

        return batch_loss, batch_accuracy

    def _get_number_of_batches(self, task_size):
        num_batches = int(np.ceil(float(task_size) / float(self.args.batch_size)))
        if num_batches > 1 and (task_size % self.args.batch_size == 1):
            num_batches -= 1

        return num_batches

    def _get_batch_indices(self, index, last_element):
        batch_start_index = index * self.args.batch_size
        batch_end_index = batch_start_index + self.args.batch_size
        if batch_end_index == (last_element - 1):  # avoid batch size of 1
            batch_end_index = last_element
        if batch_end_index > last_element:
            batch_end_index = last_element
        return batch_start_index, batch_end_index

    def validate(self):
        with torch.no_grad():
            accuracy_dict ={}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.dataset.get_validation_task(item)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    if self.use_batches:
                        self.model.build_task_representation_by_batch(context_images)
                        context_features = self._compute_features_by_batch(context_images, MetaLearningState.META_TEST)
                        self.model.configure_classifier(context_features, context_labels)
                        test_set_size = len(target_labels)
                        num_batches = self._get_number_of_batches(test_set_size)
                        target_logits = []
                        for batch in range(num_batches):
                            batch_start_index, batch_end_index = self._get_batch_indices(batch, test_set_size)
                            batch_logits = self.model.predict(target_images[batch_start_index: batch_end_index],
                                                              MetaLearningState.META_TEST)
                            target_logits.append(batch_logits)
                        target_logits = torch.vstack(target_logits)
                        target_accuracy = self.accuracy_fn(target_logits, target_labels)
                        del target_logits
                        accuracies.append(target_accuracy.item())
                    else:
                        target_logits = self.model(context_images, context_labels, target_images, MetaLearningState.META_TEST)
                        accuracy = self.accuracy_fn(target_logits, target_labels)
                        accuracies.append(accuracy.item())
                        del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
        return accuracy_dict

    def test(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Testing model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))

        with torch.no_grad():
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict = self.dataset.get_test_task(item)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    if self.use_batches:
                        self.model.build_task_representation_by_batch(context_images)
                        context_features = self._compute_features_by_batch(context_images, MetaLearningState.META_TEST)
                        self.model.configure_classifier(context_features, context_labels)
                        test_set_size = len(target_labels)
                        num_batches = self._get_number_of_batches(test_set_size)
                        target_logits = []
                        for batch in range(num_batches):
                            batch_start_index, batch_end_index = self._get_batch_indices(batch, test_set_size)
                            batch_logits = self.model.predict(target_images[batch_start_index: batch_end_index],
                                                              MetaLearningState.META_TEST)
                            target_logits.append(batch_logits)
                        target_logits = torch.vstack(target_logits)
                        target_accuracy = self.accuracy_fn(target_logits, target_labels)
                        del target_logits
                        accuracies.append(target_accuracy.item())
                    else:
                        target_logits = self.model(context_images, context_labels, target_images,
                                                   MetaLearningState.META_TEST)
                        accuracy = self.accuracy_fn(target_logits, target_labels)
                        accuracies.append(accuracy.item())
                        del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                self.logger.print_and_log('{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def _test_transfer_learning(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Testing model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))

        context_set_size = 1000
        datasets = [
            {'name': "caltech101", 'task': None, 'enabled': True},
            {'name': "cifar100", 'task': None, 'enabled': True},
            {'name': "oxford_flowers102", 'task': None, 'enabled': True},
            {'name': "oxford_iiit_pet", 'task': None, 'enabled': True},
            {'name': "sun397", 'task': None, 'enabled': True},
            {'name': "svhn_cropped", 'task': None, 'enabled': True},
            {'name': "eurosat", 'task': None, 'enabled': True},
            {'name': "resisc45", 'task': None, 'enabled': True},
            {'name': "patch_camelyon", 'task': None, 'enabled': True},
            {'name': "diabetic_retinopathy_detection", 'task': None, 'enabled': True},
            {'name': "clevr", 'task': "count", 'enabled': True},
            {'name': "clevr", 'task': "distance", 'enabled': True},
            {'name': "dsprites", 'task': "location", 'enabled': True},
            {'name': "dsprites", 'task': "orientation", 'enabled': True},
            {'name': "smallnorb", 'task': "azimuth", 'enabled': True},
            {'name': "smallnorb", 'task': "elevation", 'enabled': True},
            {'name': "dmlab", 'task': None, 'enabled': True},
            {'name': "kitti", 'task': None, 'enabled': True},
        ]

        with torch.no_grad():
            for dataset in datasets:
                if dataset['enabled'] is False:
                    continue

                if dataset['name'] == "sun397":  # use the image folder reader as the tf reader is broken for sun397
                    dataset_reader = ImageFolderReader(
                        path_to_images=self.args.download_path_for_sun397_dataset,
                        context_batch_size=context_set_size,
                        target_batch_size=self.args.batch_size,
                        image_size=self.args.image_size,
                        device=self.device)
                else:  # use the tensorflow dataset reader
                    dataset_reader = TfDatasetReader(
                        dataset=dataset['name'],
                        task=dataset['task'],
                        context_batch_size=context_set_size,
                        target_batch_size=self.args.batch_size,
                        path_to_datasets=self.args.download_path_for_tensorflow_datasets,
                        image_size=self.args.image_size,
                        device=self.device
                    )
                context_images, context_labels = dataset_reader.get_context_batch()
                self.model.build_task_representation_by_batch(context_images)
                context_features = self._compute_features_by_batch(context_images, MetaLearningState.META_TEST)
                self.model.configure_classifier(context_features, context_labels)

                test_set_size = dataset_reader.get_target_dataset_length()
                num_batches = self._get_number_of_batches(test_set_size)
                target_logits = []
                target_labels = []
                for batch in range(num_batches):
                    batch_target_images, batch_target_labels = dataset_reader.get_target_batch()
                    batch_logits = self.model.predict(batch_target_images, MetaLearningState.META_TEST)
                    target_logits.append(batch_logits)
                    target_labels.append(batch_target_labels)
                target_logits = torch.vstack(target_logits)
                target_labels = torch.hstack(target_labels)
                target_accuracy = self.accuracy_fn(target_logits, target_labels)
                del target_logits
                accuracy = target_accuracy * 100.0
                if dataset['task'] is None:
                    self.logger.print_and_log('{0:}: {1:3.1f}'.format(dataset['name'], accuracy))
                else:
                    self.logger.print_and_log('{0:} {1:}: {2:3.1f}'.format(dataset['name'], dataset['task'], accuracy))

    def _compute_features_by_batch(self, images, meta_learning_state):
        features = []
        num_images = images.size(0)
        num_batches = self._get_number_of_batches(num_images)
        for batch in range(num_batches):
            batch_start_index, batch_end_index = self._get_batch_indices(batch, num_images)
            features.append(self.model.get_context_features(images[batch_start_index: batch_end_index],
                                                            meta_learning_state))
        return torch.vstack(features)

    def _compute_features_with_split_batch(self, images, grad_indices, no_grad_indices, meta_learning_state):
        num_images = images.size(0)
        if self.feature_cache is None:    # cache the part with no gradients
            features = []
            num_batches = self._get_number_of_batches(num_images)
            for batch in range(num_batches):
                batch_start_index, batch_end_index = self._get_batch_indices(batch, num_images)
                torch.set_grad_enabled(False)
                features.append(self.model.get_context_features(images[batch_start_index: batch_end_index],
                                                                meta_learning_state))
            self.feature_cache = torch.vstack(features).to(self.device)

        # now select some random images for that will have gradients and process those
        embeddings = []
        torch.set_grad_enabled(True)
        embeddings.append(self.model.get_context_features(images[grad_indices], meta_learning_state))

        # now add in the no_grad images
        embeddings.extend(self.feature_cache[no_grad_indices])

        return torch.vstack(embeddings)

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np = shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
        }, os.path.join(self.log_files.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.log_files.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_accuracies.replace(checkpoint['best_accuracy'])


if __name__ == "__main__":
    main()
