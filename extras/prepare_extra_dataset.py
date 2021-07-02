import os
import sys
sys.path.append(os.path.abspath(os.environ['META_DATASET_ROOT']))
from meta_dataset.data import learning_spec
from meta_dataset.dataset_conversion.dataset_to_records import DatasetConverter, write_tfrecord_from_directory
from PIL import Image
from six.moves import range
import gzip
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet the TensorFlow warnings


class ExtraDatasetConverter(DatasetConverter):
    def create_splits(self):
        class_names = sorted(os.listdir(self.data_root))
        return {'train': [], 'valid': [], 'test': class_names}

    def create_dataset_specification_and_records(self):
        splits = self.get_splits(force_create=True)
        self.classes_per_split[learning_spec.Split.TRAIN] = len(splits['train'])
        self.classes_per_split[learning_spec.Split.VALID] = len(splits['valid'])
        self.classes_per_split[learning_spec.Split.TEST] = len(splits['test'])

        for class_id, class_name in enumerate(splits['test']):
            print('Creating record for class ID {} ({})'.format(class_id, class_name))
            class_directory = os.path.join(self.data_root, class_name)
            class_records_path = os.path.join(self.records_path, self.dataset_spec.file_pattern.format(class_id))
            self.class_names[class_id] = class_name
            self.images_per_class[class_id] = write_tfrecord_from_directory(class_directory, class_id, class_records_path)


def process_mnist(datasrc_path):
    def get_images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 28, 28)

    def get_labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        return integer_labels

    def create_image_dir(images, labels, path):
        if not os.path.exists(path):
            os.makedirs(path)
        class_counter = {}
        for cls in range(10):
            class_counter[cls] = 0
        for step, (image, label) in enumerate(zip(images, labels)):
            im = Image.fromarray(image)
            im = im.convert('RGB')
            im = im.resize((84, 84), resample=Image.LANCZOS)

            class_dir = os.path.join(path, 'class_%s' % label.item())
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            im_file = class_dir + '/image_%s.png' % (class_counter[label])
            im.save(im_file)
            class_counter[label] += 1
            if (step + 1) % 1000 == 0:
                print('Processed %s images.' % (step + 1))

    test_images = get_images(os.path.join(datasrc_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = get_labels(os.path.join(datasrc_path, 't10k-labels-idx1-ubyte.gz'))
    create_image_dir(test_images, test_labels, os.path.join(datasrc_path, 'mnist'))


def main():
    datasrc_path = os.path.abspath(os.environ['DATASRC'])
    records_path = os.path.abspath(os.environ['RECORDS'])
    splits_path = os.path.abspath(os.environ['SPLITS'])

    print('Processing MNIST test set.')
    process_mnist(datasrc_path)
    converter = ExtraDatasetConverter(
        name = 'mnist',
        data_root = os.path.join(datasrc_path, 'mnist'),
        has_superclasses = False,
        records_path = os.path.join(records_path, 'mnist'),
        split_file = os.path.join(splits_path, 'mnist_splits.json'),
        random_seed = 22
    )
    converter.convert_dataset()

    print('Finished.')


if __name__ == '__main__':
    main()
