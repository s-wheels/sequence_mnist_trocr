# Sequence MNIST OCR Challenge

Optical Character Recognition model to read sequences of handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in Python.

## Installation

The necessary environment can be built using Docker and the attached Dockerfile. Once Docker is installed run:

```
docker build .
```

## Directory Structure


```
sequence-mnist
│   README.md
│   submission.ipynb - Jupyter Notebook containing model training and prediction   
│
└───sequence_mnist
│   │
│   └───data
│   │   │   __init__.py
│   │   │   file_utils.py - contains utilities for downloading/loading MNIST dataset
│   │   │   mnist.py - class for loading MNIST dataset
│   │
│   └───model
│       │   __init__.py
│       │   dataloader.py - contains SequenceMNIST class for loading sequences of digits
│
└───tests
    │   __init__.py
    │   test_sequence_mnist.py - tests for the SequenceMNIST class

```

## Data

The raw dataset is MNIST which is downloaded through the Torchvision library. 

5 MNIST images with relevant number labels are appended to create the MNIST Sequence.


## Model, Training & Prediction

The model used is a pretrained Visual Encoder - Text Decoder Transformer network from [1]. I download the small version of TrOCR using Hugging Face and replace the classification head with 10 outputs (0-9).

There is a certain amount of redundancy in this network as it is explicitly pretrained for a standard OCR task of text prediction. Because of this the model is autoregressive and has an inference cost of O(n). However the normal conditional word probabilities do not apply in this instance given the fact that the numbers can be in practice considered independent random variables. A more efficient implementation would not use autoregression for this particular task and hence reduce computational complexity to O(1).

NB: Although in theory they are not random variables since the classes are imbalanced slightly the imbalance is small. Nor are they independent since I do not use sampling with replacement. But these effects should be small and not effect the non-utility of autoregression in this case (for small sequences).



### References

[1] Li et al. TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models, 2021.