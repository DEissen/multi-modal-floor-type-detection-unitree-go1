import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_Like_multimodal(nn.Module):
    """
        Model based on the LeNet-5 architecture, but with different amount of neurons in the layers.
        Paper introducing the architecture: https://ieeexplore.ieee.org/document/726791
    """

    def __init__(self, num_classes, sensors, num_input_features_dict, dropout_rate):
        """
            Init method for the LeNet_Like. It expects an 64x64 picture as input.

            Parameters:
                - num_classes (int): Number of output neurons/ classes
                - sensors (list): List of all sensors which shall be used
                - num_input_features_dict (dict): Dict containing number of input features for all sensors
                                                  (key = sensor / value = number input features) 
                - dropout_rate (float): Dropout rate for the dense layers
        """
        super(LeNet_Like_multimodal, self).__init__()

        self.sensors = sensors

        # initialize variables to calculate number of neurons for classification part
        num_cameras = 0
        num_timeseries = 0

        # ### create module dict for feature extractors
        self.feature_extractors = nn.ModuleDict()
        for sensor in self.sensors:
            if "Cam" in sensor:
                num_cameras += 1
                self.feature_extractors[sensor] = nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5),
                    nn.MaxPool2d(2, 2)
                )
            else:
                num_timeseries += 1
                # add layers for timeseries feature extractor for this sensor
                self.feature_extractors[sensor] = nn.Sequential(
                    nn.Conv1d(num_input_features_dict[sensor], 6, 5),
                    nn.MaxPool1d(2),
                    nn.Conv1d(6, 16, 5),
                    nn.MaxPool1d(2, 2)
                )

        # ### add dense layers for classification
        # calculate number of neurons for the first classification layer
        self.neurons_for_cameras = num_cameras * 16 * 13 ** 2
        self.neurons_for_timeseries = num_timeseries * 16 * 9
        # create classification layers "sub model" based on previous calculation
        self.classification_layers = nn.Sequential(
            nn.Linear(self.neurons_for_cameras +
                      self.neurons_for_timeseries, 128),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, data_struct):
        # ### get the result of each feature head
        cat_input = []  # list for inputs for feature concatenation
        for sensor in self.sensors:
            # append output/ extracted features for sensor to the concatenation list
            features_for_sensor = self.feature_extractors[sensor](data_struct[sensor])
            cat_input.append(features_for_sensor.view(features_for_sensor.size(0), -1))
        # concatenate all features
        x = torch.cat(cat_input, dim=1)

        # ### Use .view() method as Flatten Layer 
        x = x.view(-1, self.neurons_for_cameras + self.neurons_for_timeseries)

        # ### calculate classification based on concatenated features
        x = self.classification_layers(x)
        return x
