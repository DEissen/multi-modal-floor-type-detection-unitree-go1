import torch
import os

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset
from models.unimodal_models import LeNet_Like, VGG_Like, LeNet_Like1D
from models.multimodal_models import LeNet_Like_multimodal
from train import Trainer
from eval import evaluate, load_state_dict
from visualization.visualization import visualize_data_sample_or_batch
from custom_utils.custom_utils import gen_run_dir, start_logger, store_used_config

TRAINING = True


def main():
    run_paths_dict = gen_run_dir(r"")
    start_logger(run_paths_dict["logs_path"], stream_log=True)

    # variables for dataset and config to use
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_1.0"
    mapping_filename = "label_mapping_full_dataset.json"
    preprocessing_config_filename = "preprocessing_config.json"
    faulty_data_creation_config_filename = "faulty_data_creation_config.json"

    # list of sensors to use
    # sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'gyroscope',
    #            'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'rpy', 'RightCamLeft', 'RightCamRight', 'velocity', 'yawSpeed']
    sensors = ["BellyCamLeft", "accelerometer"]

    # config for training
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

    # create dataset
    transformed_dataset = FloorTypeDetectionDataset(
        dataset_path, sensors, mapping_filename, preprocessing_config_filename, faulty_data_creation_config_filename)

    # get all possible config dicts for logging
    label_mapping_dict = transformed_dataset.get_mapping_dict()
    preprocessing_config_dict = transformed_dataset.get_preprocessing_config()

    # split in train and test dataset
    train_size = int(0.9 * len(transformed_dataset))
    test_size = len(transformed_dataset) - train_size
    ds_train, ds_test = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])

    # # visualize sample from dataset for testing
    # data_for_vis, label_for_vis = ds_train.__getitem__(0)
    # visualize_data_sample_or_batch(data_for_vis, label_for_vis)

    # define model
    # ## for images
    # model = LeNet_Like(train_config_dict["num_classes"])
    # model = VGG_Like(
    #     train_config_dict["num_classes"], train_config_dict["dropout_rate"])

    # ## for IMU data
    # # determine number of input features
    # _, (training_sample, _) = next(enumerate(ds_train))
    # num_input_features = training_sample[sensors[0]].size()[0]
    # # define model
    # model = LeNet_Like1D(train_config_dict["num_classes"], num_input_features)
    
    # ## multimodal models
    # determine number of input features for all timeseries sensors
    num_input_features_dict = {}
    for sensor in sensors:
        if not "Cam" in sensor:
            _, (training_sample, _) = next(enumerate(ds_train))
            num_input_features_dict[sensor] = training_sample[sensor].size()[0]
    # define multimodal model
    model = LeNet_Like_multimodal(train_config_dict["num_classes"], sensors, num_input_features_dict)

    if TRAINING == True:
        # training loop
        trainer = Trainer(model, ds_train, ds_test, sensors,
                          train_config_dict, run_paths_dict)
        trainer.train()
    else:
        # optionally load instead of train the model
        num_ckpt = 6
        load_path = os.path.join(
            run_paths_dict["model_ckpts"], f"{model._get_name()}_{num_ckpt}.pt")
        load_state_dict(model, load_path)

    # test loop
    evaluate(model, ds_test, sensors, train_config_dict)

    # store used config as final step of logging
    store_used_config(run_paths_dict, label_mapping_dict,
                      preprocessing_config_dict, train_config_dict)


if __name__ == "__main__":
    main()
