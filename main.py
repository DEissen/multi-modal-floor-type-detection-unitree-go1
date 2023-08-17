import torch
import torchvision.transforms as transforms

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset
from unimodal_models.LeNet import LeNet
from train import Trainer
from eval import evaluate
from visualization.visualization import visualize_data_sample_or_batch
from custom_utils.custom_utils import gen_run_dir, start_logger, store_used_config

if __name__ == "__main__":
    run_paths_dict = gen_run_dir()
    start_logger(run_paths_dict["logs_path"], stream_log=True)

    # variables for dataset and config to use
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_0.1"
    mapping_filename = "label_mapping_binary.json"
    preprocessing_config_filename = "preprocessing_config.json"

    # list of sensors to use
    # sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'footRaiseHeight', 'gyroscope',
    #            'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'RightCamLeft', 'RightCamRight', 'temperature', 'velocity', 'yawSpeed']
    sensors = ["BellyCamRight"]

    # config for training
    train_config_dict = {
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.001,
        "momentum": 0.9,
        "use_wandb": False,
        "visualize_results": False
    }

    # create dataset
    transformed_dataset = FloorTypeDetectionDataset(
        dataset_path, sensors, mapping_filename, preprocessing_config_filename)

    # get all possible config dicts for logging
    label_mapping_dict = transformed_dataset.get_mapping_dict()
    preprocessing_config_dict = transformed_dataset.get_preprocessing_config()

    # split in train and test dataset
    train_size = int(0.8 * len(transformed_dataset))
    test_size = len(transformed_dataset) - train_size
    ds_train, ds_test = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])

    # # visualize sample from dataset for testing
    # data_for_vis, label_for_vis = ds_train.__getitem__(0)
    # visualize_data_sample_or_batch(data_for_vis, label_for_vis)

    # define model, loss and optimizer
    model = LeNet()

    # training loop
    trainer = Trainer(model, ds_train, ds_test, sensors,
                      train_config_dict, 50, run_paths_dict, train_config_dict["use_wandb"])
    trainer.train()

    # test loop
    evaluate(model, ds_train, sensors, train_config_dict, train_config_dict["visualize_results"])

    # store used config as final step of logging
    store_used_config(run_paths_dict, label_mapping_dict, preprocessing_config_dict, train_config_dict)