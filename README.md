# Multi-Modal_ML-Eissen (FA 3526, MA 3606)
This is the repo for the following works at the [IAS Institute from University Stuttgart](https://www.ias.uni-stuttgart.de/en/):
- FA 3526 "Robustes und Adaptives Multi-Modal Machine Learning für die Bodentyperkennung eines mobilen Roboters"
- MA 3606 "Multi-Modale Bodentyperkennung mithilfe von Transformer Netzwerken"
    
It contains the [floor-type-detection-dataset](https://github.com/DEissen/floor-type-detection-dataset) repository as a submodule for using the FTD Dataset. \
The *main.py* program must be executed for training and evaluation of models where different use cases can be run by modifying the configuration in */configs/default_config.json*.
Besides *main.py* only the programs of *hyperparameter_optimization.py* and *manual_tests.py* are intended to be directly executed (see details below). 

# How to use the code to train uni- and multimodal models?
This "How to" is further split in several subchapters. Each will describe how to act for different use cases.
### Prerequisites
1. Initialize submodule FTDDataset by using the following two commands:
    - git submodule init
    - git submodule update
2. Prepare the FTDDataset (either download an existing version or create a new one from some measurements)
### Common setup of the dataset
- Modify dataset config (path to train, val and test datasets), as well as the sensors which shall be used in the file *configs/default_config.json*
- Further config changes (preprocessing, failure data creation and label mapping) can be done in the config files of the FTDDataset submodule directly. For details see the explanation in the README.md of the [floor-type-detection-dataset](https://github.com/DEissen/floor-type-detection-dataset) repo.
### Changing the model architecture
- The model is created by the model_builder in *models/model_builder.py* which has always a feature fusion architecture consisting of three parts:
    1. modality nets (for modality specific feature extraction)
    2. fusion model (to combine the unimodal representations)
    3. classification head (for the final classification)
- Here is a visualization of the model:
![Model architecture](./models/model_architecture.png)
- The existing model architectures and parts can be found in the module **models/** where they can be modified and new parts can be added:
    - *classification_heads.py*: Contains all classification heads based on MultiModalBaseClass() to create final model with
    - *fusion_models.py*: Contains all fusion models based on FusionModelBaseClass() to create fusion model with
    - *modality_nets.py*: Contains all modality nets based on ModalityNetBaseClass() to create modality nets with
    - *model_base_classes.py*: Contains the base classes for each part of the architecture to take over some common parts (e.g. compatibility checks of input, ...) 
    - *model_builder.py*: Contains the logic to build a model based on the configuration from *configs/default_config.json*
    - *positional_encodings.py*: Contains all positional encodings for transformers
    - *tokenEmbeddings.py*: Contains all token embeddings for transformers
    - *transformers.py*: Contains all transformer layers and blocks
- Creating a new model architecture:
    - *Build model with existing parts:* Modify the "types" for each part of the model and it's parameters in the configuration *configs/default_config.json*
    - *Build model with new parts:* Implement the new parts (e.g. a new modality net) in the correct file by using the templates at the end of the file (this already considers to inherit from the base classes and provides hints for implementation) and extend the model_builder() and the configuration in *configs/default_config.json* accordingly.

### Training of a single model
For training a following parameters must be set:
- perform_training of main() must be set to True (default value!)
- sensors list should be modified to contain all sensors which the model shall use (should be done in *configs/default_config.json* and not as function parameter)
- run_path of main() should be an empty string to prevent mixing up some logs (default value!)
- num_ckpt_to_load of main() should be empty (default value!)
- logger of main() should be None (default value!)
- modify the configuration for the training in *configs/default_config.json* and not by providing a dict as the function parameter train_config_dict
### Training of multiple models
- You can train multiple models by calling the main() function multiple times. Examples can be found in *manual_tests.py* where functions are present to:
    - complete_unimodal_test(): train a model architecture for all sensors as a unimodal net
    - test_random_multimodal_models(): train a configurable amount of random multimodal models (same config for all models, but sensors are selected randomly)
    - perform_test_multiple_times(): train a model with the same sensors and the same config multiple times
    - evaluate_hyp_opt_runs(): Run evaluation of all models at runs_base_path, e.g. to get best model from hyperparameter optimization
### Perform hyperparameter optimization
Hyperparameter optimization is supported using wandb (Weights&Biases) which contains some user specific configuration which might be needed to be updated.
- For details about the sweep configuration, ... which is related to wandb, see the official [documentation](https://docs.wandb.ai/guides/sweeps).
- For modification of the hyperparameters an update in the sweep_configuration (tells wandb which parameters shall be optimized and which values these can take) dict and inside function update_config_dict_with_wandb_config() (maps the chosen config from wandb to the loaded default config) is mandatory

### Evaluation of a model
For evaluation of a model, the following parameters must be set:
- perform_training of main() must be set to False
- run_path of main() must be the path where the training was logged, e.g. r"<path_to_repo>\runs\run_05_09__15_35_41"
- num_ckpt_to_load of main() should be set to the number of the checkpoint you want to load (if None is provided, the latest found checkpoint will be used)
- logger of main() should be None
- complete configuration from training will be reused from *<run_path>\config\\* including used sensors, dataset preprocessing, ...\
- you can select whether the classification shall be visualized (plot confusion matrix and prediction for one data sample) in the training configuration with the value of "visualize_results"
=> configuration must be changed at the run_path location in case you want to change something
### Check robustness of a model
This can be done for model training or model evaluation (see previous use cases) with the following additional adaptions:
- Modify the config file *<run_path>\config\faulty_data_creation_config.json* according to your use case:
    - Set value for "create_faulty_data" to true to enable faulty data creation
    - Provide lists of sensors for which faulty data shall be created (e.g. key "Cams for guassian_noise")
    - Modify the limit/ intensity values in the config for the faulty data creation (e.g. key "offset_min")
### Check whether a model is adaptive
This is not automatically supported, as this highly dependent on the model architecture.

# Folder structure and module descriptions
This section contains a brief overview about all files in the repository. The code is structured in five modules/subfolder which contain code for different purposes.
- **configs/** \
This directory contains the default configuration for training.
    - *default_config.json:* Configuration used for training
- **custom_utils/** \
This module contains some custom utility functions used in the repository.
    - *utils.py:* Utility functions mainly for logging (including storage of logs)
- **FTDDataset/** \
This is git submodule of the [floor-type-detection-dataset](https://github.com/DEissen/floor-type-detection-dataset) repository. For more details see the README.md of the submodule.
- **metrics/** \
This module contains all code related to metrics for training/ evaluation.
    - *confusion_matrix.py:* Contains a class with methods to directly get accuracy, sensitivity, ... for two or more classes.
- **models/** \
This module contains all code related to the model architecture including the model builder where the model is built by combining the parts of the other modules.
    - *classification_heads.py*: Contains all classification heads based on MultiModalBaseClass() to create final model with
    - *fusion_models.py*: Contains all fusion models based on FusionModelBaseClass() to create fusion model with
    - *modality_nets.py*: Contains all modality nets based on ModalityNetBaseClass() to create modality nets with
    - *model_architecture.png*: Image showing model architecture for README.md
    - *model_base_classes.py*: Contains the base classes for each part of the architecture to take over some common parts (e.g. compatibility checks of input, ...) 
    - *model_builder.py*: Contains the logic to build a model based on the configuration from *configs/default_config.json*
    - *positional_encodings.py*: Contains all positional encodings for transformers
    - *tokenEmbeddings.py*: Contains all token embeddings for transformers
    - *transformers.py*: Contains all transformer layers and blocks
- **visualization/** \
This module contains a file different visualization functions.
    - *visualization.py:* Contains a function to plot data (as sample or batch) from uni or multi-modal FloorTypeDetectionDataset as well as a function to visualize the weights of the first classification layer from a model.
- *evaluation.py*: Contains a function to evaluate a trained model.
- *hyperparameter_optimization.py*: Contains a function to perform a hyperparameter optimization using Weights&Biases.
- *main.py*: Main program to start training and/ or evaluation of a model
- *manual_tests.py*: Contains functions to perform multiple trainings in a row (e.g. to train the same model multiple times)
- *README.md*: The file you are reading right now :)
- *test.ipynb*: Jupyter Notebook for testing of functions which already contains different imports for testing
- *trainer.py*: Contains a class for training a model and logging the metrics to a log file and Weights&Biases.

# Authors
- Dominik Eißen (st177975@stud.uni-stuttgart.de)