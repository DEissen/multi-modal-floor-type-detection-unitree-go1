import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_Like_multimodal(nn.Module):
    """
        Model based on the LeNet-5 architecture, but with different amount of neurons in the layers.
        Paper introducing the architecture: https://ieeexplore.ieee.org/document/726791
    """

    def __init__(self, num_classes, sensors, num_input_features_dict):
        """
            Init method for the LeNet_Like. It expects an 64x64 picture as input.

            Parameters:
                - num_classes (int): Number of output neurons/ classes.
                - sensors (list): List of all sensors which shall be used. 
        """
        super(LeNet_Like_multimodal, self).__init__()

        self.sensors = sensors

        # variables to calculate number of neurons for classification part
        num_cameras = 0
        num_timeseries = 0

        for sensor in self.sensors:
            if "Cam" in sensor:
                num_cameras += 1
                self.camera_feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5),
                    nn.MaxPool2d(2, 2)
                )
            else:
                num_timeseries += 1
                # add layers for timeseries feature extractor for this sensor
                self.timeseries_feature_extractor = nn.Sequential(
                    nn.Conv1d(num_input_features_dict[sensor], 6, 5),
                    nn.MaxPool1d(2),
                    nn.Conv1d(6, 16, 5),
                    nn.MaxPool1d(2, 2)
                )

        # add dense layers for classification
        self.neurons_for_cameras = num_cameras * 16 * 13 * 13
        self.neurons_for_timeseries = num_timeseries * 16 * 9
        self.classification_layers = nn.Sequential(
            nn.Linear(self.neurons_for_cameras + self.neurons_for_timeseries, 128),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes)
        )

    def multimodal_feature_extractor(self, data_struct):
        for sensor in self.sensors:
            if "Cam" in sensor:
                x = self.camera_feature_extractor(data_struct[sensor])
            else:
                y = self.timeseries_feature_extractor(data_struct[sensor])
        x = torch.cat((x.view(x.size(0), -1), y.view(y.size(0), -1)), dim=1)
        return x

    def forward(self, data_struct):
        x = self.multimodal_feature_extractor(data_struct)
        x = x.view(-1, self.neurons_for_cameras + self.neurons_for_timeseries)
        x = self.classification_layers(x)
        return x
