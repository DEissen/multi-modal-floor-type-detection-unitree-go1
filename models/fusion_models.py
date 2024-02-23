import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import FusionModelBaseClass


class Concatenate_FusionModel(FusionModelBaseClass):
    """
        Class for simple feature fusion with concatenation of flatten outputs.
    """

    def __init__(self, sensors, model_config_dict, modality_nets, sample_batch):
        """
            Init method of ConcatenateForFusion() class.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - model_config_dict (dict): Dict which contains all configuration parameters for the model
                - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensors, model_config_dict, modality_nets)

        # #### calculate shape of output of the fusion model based on sample batch
        self.calculate_features_from_sample_batch(sample_batch)

    def forward(self, data_dict):
        """
            Implementation of forward() method to process data.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Returns:
                - x (torch.Tensor): Representation vector after processing the data
        """
        # ### get the result of each feature head
        cat_input = []  # list for inputs for feature concatenation
        for sensor in self.sensors:
            # append output/ extracted features for sensor to the concatenation list
            features_for_sensor = self.modality_nets[sensor](
                data_dict)
            # ### Use .view() method as Flatten Layer
            cat_input.append(features_for_sensor.view(
                features_for_sensor.size(0), -1))

        # ### concatenate all features
        x = torch.cat(cat_input, dim=1)

        return x


# class Template_FusionModel(FusionModelBaseClass):
#     """
#          Template fusion model as baseline for new modality nets.
#     """

#     def __init__(self, sensors, model_config_dict, modality_nets, sample_batch):
#         """
#             Init method of Template_FusionModel() class.

#             Parameters:
#                 - sensor (string): Name of the sensor for which this modality net is used
#                 - model_config_dict (dict): Dict which contains all configuration parameters for the model
#                 - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
#                 - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
#         """
#         # #### call init method of superclass
#         super().__init__(sensors, model_config_dict, modality_nets)

#         # #### define layers
#         # TODO: add layers as members

#         # #### calculate shape of output of the fusion model based on sample batch
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
