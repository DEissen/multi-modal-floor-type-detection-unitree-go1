import torch
import torchvision.transforms as transforms

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset
from unimodal_models.LeNet import LeNet
from train import Trainer
from eval import evaluate
from visualization.visualization import visualize_data_sample_or_batch

if __name__ == "__main__":
    # variables for dataset and config to use
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_0.1"
    mapping_filename = "label_mapping_binary.json"
    preprocessing_config_filename = "preprocessing_config.json"

    # list of sensors to use
    # sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'footRaiseHeight', 'gyroscope',
    #            'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'RightCamLeft', 'RightCamRight', 'temperature', 'velocity', 'yawSpeed']
    sensors = ["BellyCamRight"]

    # config for training
    train_config = {
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.001,
        "momentum": 0.9
    }

    # create dataset
    transformed_dataset = FloorTypeDetectionDataset(
        dataset_path, sensors, mapping_filename, preprocessing_config_filename)

    # split in train and test dataset
    train_size = int(0.8 * len(transformed_dataset))
    test_size = len(transformed_dataset) - train_size
    ds_train, ds_test = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])

    # visualize sample from dataset for testing
    data_for_vis, label_for_vis = ds_train.__getitem__(0)
    visualize_data_sample_or_batch(data_for_vis, label_for_vis)

    # define model, loss and optimizer
    model = LeNet()

    # training loop
    trainer = Trainer(model, ds_train, ds_test, sensors,
                      train_config, 50, None, True)
    trainer.train()

    # test loop
    evaluate(model, ds_train, sensors, train_config, True)
