# Multi-Modal_ML-Eissen (FA 3526)
This is the repo for FA 3526 "Robustes und Adaptives Multi-Modal Machine Learning für die Bodentyperkennung eines mobilen Roboters". It contains the [Floor-Type-Detection-Dataset](https://github.tik.uni-stuttgart.de/ac136427/Floor-Type-Detection-Dataset) repository as a submodule for using the FTD Dataset. \
Only the *main.py* program must be executed for training, evaluation, ... where the different use cases can be run by modifying the configuration of the program.

# How to use the code to train uni- and multimodal models?
This "How to" is further split in several subchapters. Each will describe how to act for different use cases.
### Prerequisites
1. Initialize submodule FTDDataset by using the following two commands:
    - git submodule init
    - git submodule update
2. Prepare your version of the FTDDataset (either download an existing version or create your own from some measurements)
### Training of a model
1. TODO
### Evaluation of a model
1. TODO
### Check robustness of a model
1. TODO
### Check whether a model is adaptive
1. TODO

# Folder structure and module descriptions
This section contains a brief overview about all files in the repository. The code is structured in four modules/subfolder which contain code for different purposes.
- **custom_utils/** \
This module contains some custom utility functions used in the repository.
    - *custom_utils.py:* Utility functions mainly for logging (including storage of logs)
- **FTDDataset/** \
This is git submodule of the [Floor-Type-Detection-Dataset](https://github.tik.uni-stuttgart.de/ac136427/Floor-Type-Detection-Dataset) repository. For more details see the README.md of the submodule.
- **metrics/** \
This module contains all code related to metrics for training/ evaluation.
    - *confusion_matrix.py:* Contains a class with methods to directly get accuracy, sensitivity, ... for two or more classes.
- **models/** \
This module contains the used model architectures.
    - *multimodal_models.py:* Contains a class to create a multimodal model with two or more sensors from the FTD Dataset as input.
    - *unimodal_models.py:* Contains two models for images and one model for timeseries data as input.
- **visualization/** \
This module contains a file with an adaptive visualization function for different kinds of data.
    - *visualization.py:* Contains a function to plot data (as sample or batch) from uni or multi-modal FloorTypeDetectionDataset.
- *eval.py*: Contains a function to evaluate a trained model.
- *main.py*: Main program to start training and/ or evaluation of a model
- *README.md*: The file you are reading right now :)
- *train.py*: Contains a class for training a model and logging the metrics to a log file and Weights&Biases.

# Authors
- Dominik Eißen (st177975@stud.uni-stuttgart.de)