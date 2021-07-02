# LITE: Memory Efficient Meta-Learning with Large Images
This repository contains the code to reproduce the VTAB+MD few-shot classification experiments carried out in:
"Memory Efficient Meta-Learning with Large Images". The code for the ORBIT experiments can be found
[here](https://github.com/microsoft/ORBIT-Dataset).

## Dependencies
This code requires the following:
* Python 3.7 or greater
* PyTorch 1.8 or greater (most of the code is written in PyTorch)
* TensorFlow 2.3 or greater (for reading Meta-Dataset datasets and VTAB datasets)
* TensorFlow Datasets 4.3 or greater (for reading VTAB datasets)
* Gin Config 0.4 or greater (needed for the Meta-Dataset reader)

## GPU Requirements
* To reproduce the results in the paper by meta-training, a GPU with 16GB of memory is required. By reducing the batch
  size, it is possible to run on a GPU with less memory, but classification results may be different.

## Installation
The following steps will take a considerable length of time and disk space.
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset)
      for "Installation" and "Downloading and converting datasets".
3. Install additional training dataset (MNIST):
    * Change to the $DATASRC directory: ```cd $DATASRC```
    * Download the MNIST test images: ```wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz```
    * Download the MNIST test labels: ```wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz```
    * Change to the ```extras``` directory in the repository.
    * Run: ```python prepare_extra_dataset.py```
4. Update ILSVRC_2012 and MNIST dataset_spec files:
   * Replace the file $RECORDS/ilsvrc_2012/dataset_spec.json with the one in the extras/ilsvrc_2012 directory in
     this repo. This change will allow training on all 1000 classes of ilsvrc_2012 as is permitted with MD-v2.
   * Replace the file $RECORDS/mnist/dataset_spec.json with the one in the extras/mnist directory in this repo.
   This will convert MINST into a training dataset as is permitted with MD-v2.
5. The VTAB-v2 benchmark uses [TensorFlow Datasets](https://www.tensorflow.org/datasets). The majority of these are
   downloaded and pre-processed upon first use. However, the 
   [Diabetic Retinopathy](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection) 
   and [Resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) datasets need to be
   downloaded manually. Click on the links for details. 
6. Tensorflow Datasets has a [bug](https://github.com/tensorflow/datasets/issues/2889) where it will crash when 
   installing the sun397 dataset that is a part of VTAB-v2. As a workaround, the sun397 dataset needs to be 
   installed manually:
   * Change to the directory of your choice and download and extract the sun397 dataset from the following URL:
   ```https://drive.google.com/file/d/1yByBXgYbNKVitBalOwbwJ1LlVg6Vzreg/view?usp=sharing``` 
   

## Usage
To train and test on VTAB+MD:

1. First run the following two commands.

   ```ulimit -n 50000```

   ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```

   Note the above commands need to be run every time you open a new command shell.

2. Then switch to the ```src``` directory in this repo and execute any of the following pairs command lines.
   The first meta-trains and meta-tests on MD-v2 and the second meta-tests on VTAB-v2. 

   **<ins>LITE and 224x224 image size<ins>**:
     
   Meta-train and meta-test on MD-v2: 
   
   ```python run.py --data_path <path to meta-dataset records> -i 10000 --batch_size 40 -c <path to checkpoint directory>```

   Meta-Test on VTAB-v2:
   
   ```python run.py --batch_size 40 -c <path to checkpoint directory> --mode test_vtab --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded> --download_path_for_sun397_dataset <path to sun397 images> -m <path to model to test>```

   **<ins>No LITE, Small Task Size, and 224x224 image size<ins>**:

   Meta-train and meta-test on MD-v2:

   ```python run.py --data_path <path to meta-dataset records> --train_method small_task -i 15000 --max_support_train 40 --max_way_train 30 --batch_size 40 -c <path to checkpoint directory>```

   Meta-Test on VTAB-v2:
   
   ```python run.py --batch_size 40 -c <path to checkpoint directory> --mode test_vtab --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded> --download_path_for_sun397_dataset <path to sun397 images> -m <path to model to test>```

   **<ins>No LITE and 84x84 image size<ins>**:

   Meta-train and meta-test on MD-v2:

   ```python run.py --data_path <path to meta-dataset records> --train_method no_lite -i 35000 --image_size 84 --batch_size 501 --max_support_train 300 --max_way_train 40 -c <path to checkpoint directory>```

   Meta-Test on VTAB-v2:

   ```python run.py --image_size 84 --batch_size 501 -c <path to checkpoint directory> --mode test_vtab --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded> --download_path_for_sun397_dataset <path to sun397 images> -m <path to model to test>```

   **<ins>Test on VTAB+MD using the meta-trained model (LITE on 224x224 images) that was used to generate the results reported in the paper<ins>**:

   Meta-Test on MD-v2:
   
   ```python run.py --data_path <path to meta-dataset records> --batch_size 40 -c <path to existing checkpoint directory> --mode test -m ../models/meta-trained_lite_224.pt```

   Meta-Test on VTAB-v2:

   ```python run.py --batch_size 40 -c <path to existing checkpoint directory> --mode test_vtab --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded> --download_path_for_sun397_dataset <path to sun397 images> -m ../models/meta-trained_lite_224.pt```

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

