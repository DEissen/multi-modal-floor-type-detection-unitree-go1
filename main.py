import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# custom imports
from FTDDataset.FTDDataset import FloorTypeDetectionDataset, FTDD_Crop, FTDD_Normalize, FTDD_Rescale, FTDD_ToTensor
from unimodal_models.LeNet import LeNet

if __name__ == "__main__":
    # variables for dataset and config to use
    dataset_path = r"C:\Users\Dominik\Downloads\FTDD_0.1"
    mapping_filename = "label_mapping_binary.json"
    preprocessing_config_filename = "preprocessing_config.json"

    # list of sensors to use
    # sensors = ["accelerometer", "BellyCamRight", "BellyCamLeft", "ChinCamLeft",
    #            "ChinCamRight", "HeadCamLeft", "HeadCamRight", "LeftCamLeft", "LeftCamRight", "RightCamLeft", "RightCamRight"]
    sensors = ["BellyCamRight"]

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
    train_dataset, test_dataset = torch.utils.data.random_split(
        transformed_dataset, [train_size, test_size])

    # create dataloader
    trainloader = DataLoader(train_dataset,
                            batch_size=8, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset,
                            batch_size=8, shuffle=True, drop_last=True)
    
    # define model, loss and optimizer
    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training loop
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = inputs[sensors[0]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 0:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    
    print('Finished Training')

    # test loop
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images[sensors[0]]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(test_dataset)} test images: %d %%' % (
        100 * correct / total))