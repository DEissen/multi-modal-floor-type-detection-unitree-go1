import torch
import torchvision.transforms as transforms

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset, FTDD_Crop, FTDD_Normalize, FTDD_Rescale, FTDD_ToTensor
from unimodal_models.LeNet import LeNet
from train import Trainer
from eval import evaluate

if __name__ == "__main__":
    # variables for dataset and config to use
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_0.1"
    mapping_filename = "label_mapping_binary.json"
    preprocessing_config_filename = "preprocessing_config.json"

    # list of sensors to use
    # sensors = ["accelerometer", "BellyCamRight", "BellyCamLeft", "ChinCamLeft",
    #            "ChinCamRight", "HeadCamLeft", "HeadCamRight", "LeftCamLeft", "LeftCamRight", "RightCamLeft", "RightCamRight"]
    sensors = ["BellyCamRight"]

    # config for training
    train_config = {
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.001,
        "momentum": 0.9
    }

    # create list of transformations to perform (data preprocessing + failure case creation)
    transformations_list = []
    transformations_list.append(FTDD_Crop(preprocessing_config_filename))
    transformations_list.append(FTDD_Rescale(preprocessing_config_filename))
    transformations_list.append(FTDD_Normalize(preprocessing_config_filename))
    transformations_list.append(FTDD_ToTensor())
    composed_transforms = transforms.Compose(transformations_list)

    # create dataset
    transformed_dataset = FloorTypeDetectionDataset(
        dataset_path, sensors, mapping_filename, transform=composed_transforms)

    # split in train and test dataset
    train_size = int(0.8 * len(transformed_dataset))
    test_size = len(transformed_dataset) - train_size
    ds_train, ds_test = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])
    
    # define model, loss and optimizer
    model = LeNet()

    # training loop
    trainer = Trainer(model, ds_train, ds_test, sensors, train_config, 50, None, True)
    trainer.train()

    # test loop
    evaluate(model, ds_train, sensors, train_config)