import os
import glob

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset
from models.model_builder import model_builder
from trainer import Trainer
from evaluation import evaluate, load_state_dict
from visualization.visualization import visualize_data_sample_or_batch, visualize_weights_of_dense_layer
from custom_utils.utils import gen_run_dir, CustomLogger, store_used_config, load_run_path_config


def main(perform_training=True, sensors=None, run_path=r"", num_ckpt_to_load=None, logger=None):
    """
        Main function for training and evaluation of unimodal and multimodal models for the FTD dataset.
        Most important config parameters must be changed in the function itself, as this function is the core of the framework.
        There are a few function parameters to enable execution of subsequent runs with the same config for different sensors, ...

        Parameters:
            - perform_training (bool): Boolean that determines whether model is newly trained or whether checkpoint shall be restored.
                                       NOTE: If set to False, there must be a path provided in parameter run_path where a checkpoint is present!
            - sensors (list): List of sensors to use from the FTD dataset and for the model. Based on this the model architecture will be selected (either unimodal or multimodal)
            - run_path (str): Path to files stored from a previous run for loading a checkpoint, ...
            - num_ckpt_to_load (int): Number of the checkpoint to restore from run_path. If None is provided, the last element of the sorted file list will be taken.
            - logger (CustomLogger): Instance of custom logger class which must be provided in case main() is called multiple times by another function/program to prevent multiple logging
    """
    # ####### configurable parameters #######
    # ### variables for dataset config
    mapping_filename = "label_mapping_full_dataset.json"
    preprocessing_config_filename = "preprocessing_config.json"
    faulty_data_creation_config_filename = "faulty_data_creation_config.json"

    # ### load config based on run_path
    train_config_dict = load_run_path_config(run_path)
    train_dataset_path = train_config_dict["train_dataset_path"]
    test_dataset_path = train_config_dict["test_dataset_path"]
    if sensors == None:
        # use sensors from default config if no sensors were provided as function parameter
        sensors = train_config_dict["sensors"]
    else:
        # otherwise use sensors from function parameter and log them in config dict
        train_config_dict["sensors"] = sensors

    # ####### start of program (only modify code below if you want to change some behavior) #######
    run_paths_dict = gen_run_dir(run_path)
    # create new logger object if non was provided
    if logger == None:
        logger = CustomLogger()

    logger.start_logger(run_paths_dict["logs_path"], stream_log=True)

    # ####### prepare datasets #######
    # load datasets
    ds_train = FloorTypeDetectionDataset(
        train_dataset_path, sensors, mapping_filename, preprocessing_config_filename)
    ds_test = FloorTypeDetectionDataset(
        test_dataset_path, sensors, mapping_filename, preprocessing_config_filename, faulty_data_creation_config_filename)  # only test dataset can contain faulty data

    # ####### define model (automatically select uni or multimodal model) #######
    model = model_builder(sensors, ds_train, train_config_dict)

    # ####### start training if selected, otherwise load stored model #######
    if perform_training:
        # training loop
        trainer = Trainer(model, ds_train, ds_test, sensors,
                          train_config_dict, run_paths_dict)
        trainer.train()
    else:
        # load trained model if no training shall be done
        if num_ckpt_to_load == None:
            # if no checkpoint is provided, the last present ckpt is taken
            glob_pattern = os.path.join(run_paths_dict["model_ckpts"], "*.pt")
            ckpt_files = glob.glob(glob_pattern)
            ckpt_files.sort()
            load_path = ckpt_files[-1]
        else:
            # create load path from provided checkpoint number
            load_path = os.path.join(
                run_paths_dict["model_ckpts"], f"{model._get_name()}_{num_ckpt_to_load}.pt")

        load_state_dict(model, load_path)

        # uncomment if you want to see a visualization of the weights of the first Dense Layer
        # visualize_weights_of_dense_layer(model, sensors, False)

    # ####### start evaluation of trained/ loaded model #######
    # test loop
    evaluate(model, ds_test, sensors, train_config_dict)

    # ####### store remaining logs #######
    # store used config as final step of logging
    store_used_config(run_paths_dict, ds_train.get_mapping_dict(),
                      ds_train.get_preprocessing_config(), train_config_dict, ds_test.get_preprocessing_config())


if __name__ == "__main__":
    main()
