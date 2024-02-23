import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import ModalityNetBaseClass


class LeNet2dLike_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based on the LeNet-5 architecture. 
        Paper introducing the architecture: https://ieeexplore.ieee.org/document/726791
    """

    def __init__(self, sensor, model_config_dict, sample_batch):
        """
            Init method of LeNet2dLike_ModalityNet() class.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - model_config_dict (dict): Dict which contains all configuration parameters for the model
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, model_config_dict)

        # #### check whether input is compatible
        self.confirm_input_is_an_image(sample_batch)

        # #### define layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # #### calculate shape of output of the modality net based on sample batch
        self.calculate_features_from_sample_batch(sample_batch)

    def forward(self, data_dict):
        """
            Implementation of forward() method to process data.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Returns:
                - x (torch.Tensor): Representation vector after processing the data
        """
        x = self.pool(F.relu(self.conv1(data_dict[self.sensor])))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class LeNet1dLike_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based on the LeNet-5 architecture but with 1D Conv layers.
        Paper introducing the architecture: https://ieeexplore.ieee.org/document/726791
    """

    def __init__(self, sensor, model_config_dict, sample_batch):
        """
            Init method for the LeNet1dLike_ModalityNet.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - model_config_dict (dict): Dict which contains all configuration parameters for the model
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, model_config_dict)

        # #### check whether input is compatible
        self.confirm_input_is_timeseries_data(sample_batch)

        # #### get number of input features for time series data
        num_input_features = sample_batch[sensor].shape[1]

        # #### define layers
        self.conv1 = nn.Conv1d(num_input_features, 6, 5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(6, 16, 5)

        # #### calculate shape of output of the modality net based on sample batch
        self.calculate_features_from_sample_batch(sample_batch)

    def forward(self, data_dict):
        """
            Implementation of forward() method to process data.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Returns:
                - x (torch.Tensor): Representation vector after processing the data
        """
        x = self.pool(F.relu(self.conv1(data_dict[self.sensor])))
        x = self.pool(F.relu(self.conv2(x)))
        return x

# class Template_ModalityNet(ModalityNetBaseClass):
#     """
#         Template modality net as baseline for new modality nets.
#     """

#     def __init__(self, sensor, model_config_dict, sample_batch):
#         """
#             Init method for the Template_ModalityNet.

#             Parameters:
#                 - sensor (string): Name of the sensor for which this modality net is used
#                 - model_config_dict (dict): Dict which contains all configuration parameters for the model
#                 - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
#         """
#         # #### call init method of superclass
#         super().__init__(sensor, model_config_dict)

#         # #### check whether input is compatible
#         # TODO: uncomment wanted check and remove the other one
#         # self.confirm_input_is_timeseries_data(sample_batch)
#         # self.confirm_input_is_an_image(sample_batch)

#         # #### get number of input features for time series data
#         # TODO: uncomment if needed for time series data and remove lines completely otherwise
#         # num_input_features = sample_batch[sensor].shape[1]

#         # #### define layers
#         # TODO: add layers as members

#         # #### calculate shape of output of the modality net based on sample batch
#         self.calculate_features_from_sample_batch(sample_batch)

#     def forward(self, data_dict):
#         """
#             Implementation of forward() method to process data.

#             Parameters:
#                 - data_dict (torch.Tensor): Batch from dataset to process

#             Returns:
#                 - x (torch.Tensor): Representation vector after processing the data
#         """
#         # TODO: replace with real forward path
#         x = data_dict[self.sensor]
#         return x
