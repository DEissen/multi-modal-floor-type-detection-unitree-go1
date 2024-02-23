import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import MultiModalBaseModel


class Dense_ClassificationHead(MultiModalBaseModel):
    """
        Class for dense layer based classification head.
    """

    def __init__(self, num_classes, sensors, model_config_dict, fusion_model):
        """
            Init method of Dense_ClassificationHead() class.

            Parameters:
                - num_classes (int): Number of output neurons/ classes.
                - sensors (list): List of all sensors which shall be used
                - model_config_dict (dict): Dict which contains all configuration parameters for the model
                - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass
        """
        # #### call init method of superclass 
        super().__init__(num_classes, sensors, model_config_dict, fusion_model)

        # #### Check whether fusion model is compatible with this classification head
        fusion_model_output_shape = self.fusion_model.get_shape_output_features()
        if len(fusion_model_output_shape) > 1:
            raise TypeError("Fusion Model is not compatible, as the Dense_Classification_Head only accepts a flatten input!")

        # #### define layers
        self.fc1 = nn.Linear(int(fusion_model_output_shape[0]), 128)
        self.dropout = nn.Dropout1d(model_config_dict["dropout_rate"])
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, data_dict):
        """
            Implementation of forward() method to process data.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Returns:
                - x (torch.Tensor): Representation vector after processing the data
        """
        # #### get representations from fusion model
        x = self.fusion_model(data_dict)
        
        # #### forward path of classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class Template_ClassificationHead(MultiModalBaseModel):
#     """
#         Template classification head as baseline for new classification heads.
#     """

#     def __init__(self, num_classes, sensors, model_config_dict, fusion_model):
#         """
#             Init method of Template_ClassificationHead() class.

#             Parameters:
#                 - num_classes (int): Number of output neurons/ classes.
#                 - sensors (list): List of all sensors which shall be used
#                 - model_config_dict (dict): Dict which contains all configuration parameters for the model
#                 - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass
#         """
#         # #### call init method of superclass 
#         super().__init__(num_classes, sensors, model_config_dict, fusion_model)

#         # #### Check whether fusion model is compatible with this classification head
#         fusion_model_output_shape = self.fusion_model.get_shape_output_features()
#         # TODO: add check based on fusion_model_output_shape if necessary and possible and remove lines completely otherwise

#         # #### define layers
#         # TODO: add layers as members

#     def forward(self, data_dict):
#         """
#             Implementation of forward() method to process data.

#             Parameters:
#                 - data_dict (torch.Tensor): Batch from dataset to process

#             Returns:
#                 - x (torch.Tensor): Representation vector after processing the data
#         """
#         # #### get representations from fusion model
#         x = self.fusion_model(data_dict)
        
#         # #### forward path of classification head
#         # TODO: replace with real forward path of classification head based on x

#         return x