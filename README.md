# Sequence MNIST OCR Challenge

Optical Character Recognition model to read sequences of handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in Python.


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