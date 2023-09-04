import torch
import os
import glob

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset
from models.unimodal_models import LeNet_Like, VGG_Like, LeNet_Like1D
from models.multimodal_models import LeNet_Like_multimodal
from train import Trainer
from eval import evaluate, load_state_dict
from visualization.visualization import visualize_data_sample_or_batch, visualize_weights_of_dense_layer
from custom_utils.custom_utils import gen_run_dir, start_logger, store_used_config


def main(perform_training=True, sensors=None, run_path=r"", num_ckpt_to_load=None):
    # ####### configurable parameters #######
    # ### variables for dataset config
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_1.0"
    mapping_filename = "label_mapping_full_dataset.json"
    preprocessing_config_filename = "preprocessing_config.json"
    faulty_data_creation_config_filename = "faulty_data_creation_config.json"

    # ### create list of sensors to use if not provided by caller
    # List of all sensors available: 'accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'gyroscope',
    #                                'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'rpy', 'RightCamLeft', 'RightCamRight', 'velocity', 'yawSpeed'
    if sensors == None:
        sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft']

    # ### config for training
    train_config_dict = {
        "epochs": 2,
        "batch_size": 8,
        "optimizer": "adam",
        "lr": 0.001,
        "momentum": 0.9,
        "dropout_rate": 0.2,
        "num_classes": 4,
        "use_wandb": True,
        "visualize_results": True,
        "train_log_interval": 200,
        "sensors": sensors,
        "dataset_path": dataset_path
    }

    # ####### start of program (only modify code below if you want to change some behavior) #######
    run_paths_dict = gen_run_dir(run_path)
    start_logger(run_paths_dict["logs_path"], stream_log=True)

    # ####### prepare dataset and load config #######
    # create dataset
    transformed_dataset = FloorTypeDetectionDataset(
        dataset_path, sensors, mapping_filename, preprocessing_config_filename, faulty_data_creation_config_filename)

    # get all possible config dicts for logging
    label_mapping_dict = transformed_dataset.get_mapping_dict()
    preprocessing_config_dict = transformed_dataset.get_preprocessing_config()

    # split in train and test dataset
    if perform_training:
        train_size = int(0.9 * len(transformed_dataset))
    else:
        train_size = 0

    test_size = len(transformed_dataset) - train_size
    ds_train, ds_test = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])

    # ####### define model (automatically select uni or multimodal model) #######
    # ## multimodal model when more than 1 sensor is provided
    if len(sensors) > 1:
        # determine number of input features for all timeseries sensors
        num_input_features_dict = {}
        for sensor in sensors:
            if not "Cam" in sensor:
                _, (training_sample, _) = next(enumerate(transformed_dataset))
                num_input_features_dict[sensor] = training_sample[sensor].size()[
                    0]

        # define multimodal model
        model = LeNet_Like_multimodal(
            train_config_dict["num_classes"], sensors, num_input_features_dict, train_config_dict["dropout_rate"])

    # ## unimodal model for images if "Cam" in the sensor name
    elif "Cam" in sensors[0]:
        # for images
        model = LeNet_Like(train_config_dict["num_classes"])
        # model = VGG_Like(
        #     train_config_dict["num_classes"], train_config_dict["dropout_rate"])

    # ## unimodal model for timeseries data in remaining case
    else:
        # determine number of input features
        _, (training_sample, _) = next(enumerate(transformed_dataset))
        num_input_features = training_sample[sensors[0]].size()[0]

        # define model
        model = LeNet_Like1D(
            train_config_dict["num_classes"], num_input_features)

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
    store_used_config(run_paths_dict, label_mapping_dict,
                      preprocessing_config_dict, train_config_dict)


if __name__ == "__main__":
    main()
