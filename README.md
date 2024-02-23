# Multi-Modal_ML-Eissen (FA 3526)
This is the repo for FA 3526 "Robustes und Adaptives Multi-Modal Machine Learning für die Bodentyperkennung eines mobilen Roboters". It contains the [Floor-Type-Detection-Dataset](https://github.tik.uni-stuttgart.de/ac136427/Floor-Type-Detection-Dataset) repository as a submodule for using the FTD Dataset. \
Only the *main.py* program must be executed for training and evaluation of models where different use cases can be run by modifying the configuration of the program.
TODO: Mention sweeps_and_test.py

# How to use the code to train uni- and multimodal models?
This "How to" is further split in several subchapters. Each will describe how to act for different use cases. For each of these use cases the file *main.py* is the starting point, where you have to modify the function parameters of the main() function and the as "configurable" marked variables in the main() function accordingly.
### Prerequisites
1. Initialize submodule FTDDataset by using the following two commands:
    - git submodule init
    - git submodule update
2. Prepare your version of the FTDDataset (either download an existing version or create your own from some measurements)
### Common setup of the dataset
- Modify variables for dataset config (path to dataset and names of config files), as well as the sensors you want to use in the *configs/default_config.json* file
- Further config changes must be done in the selected config files of the FTDDataset submodule directly
### Changing the model architecture
TODO: Update section when model builder implementation is finished with Transformers
- There are three different kinds of model architectures always usable in the main() function, which should be always present:
    - A unimodal model for images (inside elif statement in line 108)
    - A unimodal model for timeseries data (inside else statement in line 114)
    - A multimodal model (inside if statement in line 94)
- The existing model architectures can be found in the module **models/** where they can be modified
- If you want to create a new model architecture you should do the following
    1. Define your model in the module **models/** similar to the existing one (important to have same input handling in order to prevent an update of the training routine)
    2. Import you model in *main.py* and replace the fitting model definition (unimodal or multimodal) with your model.

### Training of a single model
For training a following parameters must be set:
- perform_training of main() must be set to True
- sensors list should be modified for your use case (should be done in *configs/default_config.json* and not as function parameter)
- run_path of main() should be an empty string to prevent mixing up some logs
- num_ckpt_to_load of main() is not relevant
- logger of main() should be None
- modify the configuration for your use case in *configs/default_config.json*
### Training of multiple models
- TODO
### Evaluation of a model
For training a following parameters must be set:
- perform_training of main() must be set to False
- run_path of main() must be the path where the training was logged, e.g. r"D:\<path_to_repo>\runs\run_05_09__15_35_41"
- num_ckpt_to_load of main() should be set to the number of the checkpoint you want to load (if None is provided, the latest found checkpoint will be used)
- logger of main() should be None
- complete configuration from training will be reused from *<run_path>\config\\* including used sensors, dataset preprocessing, ...\
=> modify configuration there in case you want to change something
### Check robustness of a model
This can be done for model training or model evaluation (see previous use cases) with the following additional adaptions:
- Modify the config file (default config found in *FTDDataset\configs\faulty_data_creation_config.json* or in *<run_path>\config\faulty_data_creation_config.json*) according to your use case:
    - Set value for "create_faulty_data" to true to enable faulty data creation
    - Provide lists of sensors for which faulty data shall be created (e.g. key "Cams for brightness")
    - Modify the limit/ intensity values in the config for the faulty data creation (e.g. key "offset_min")
### Check whether a model is adaptive
This is not automatically supported, as this highly dependent on the model architecture.

# Folder structure and module descriptions
This section contains a brief overview about all files in the repository. The code is structured in five modules/subfolder which contain code for different purposes.
- **custom_utils/** \
This module contains some custom utility functions used in the repository.
    - *utils.py:* Utility functions mainly for logging (including storage of logs)
- **FTDDataset/** \
This is git submodule of the [Floor-Type-Detection-Dataset](https://github.tik.uni-stuttgart.de/ac136427/Floor-Type-Detection-Dataset) repository. For more details see the README.md of the submodule.
- **metrics/** \
This module contains all code related to metrics for training/ evaluation.
    - *confusion_matrix.py:* Contains a class with methods to directly get accuracy, sensitivity, ... for two or more classes.
- **models/** \
TODO: update if implemenation is finished => This module contains the used model architectures.
    - *multimodal_models.py:* Contains a class to create a multimodal model with two or more sensors from the FTD Dataset as input.
    - *unimodal_models.py:* Contains two models for images and one model for timeseries data as input.
- **visualization/** \
This module contains a file different visualization functions.
    - *visualization.py:* Contains a function to plot data (as sample or batch) from uni or multi-modal FloorTypeDetectionDataset as well as a function to visualize the weights of the first classification layer from a model.
- *evaluation.py*: Contains a function to evaluate a trained model.
- *main.py*: Main program to start training and/ or evaluation of a model
- *README.md*: The file you are reading right now :)
- TODO: Update with other files present in final version
- *sweeps_and_tests.py*: TODO when implementation is done
- *trainer.py*: Contains a class for training a model and logging the metrics to a log file and Weights&Biases.

# Authors
- Dominik Eißen (st177975@stud.uni-stuttgart.de)