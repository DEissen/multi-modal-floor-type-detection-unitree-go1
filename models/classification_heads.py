import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import MultiModalBaseClass


class Dense_ClassificationHead(MultiModalBaseClass):
    """
        Class for dense layer based classification head.
    """

    def __init__(self, num_classes, sensors, classification_head_config_dict, fusion_model):
        """
            Init method of Dense_ClassificationHead() class.

            Parameters:
                - num_classes (int): Number of output neurons/ classes.
                - sensors (list): List of all sensors which shall be used
                - classification_head_config_dict (dict): Dict containing the classification head specific configuration parameters
                - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass
        """
        # #### call init method of superclass 
        super().__init__(num_classes, sensors, classification_head_config_dict, fusion_model)
        self.num_hidden_layers = classification_head_config_dict["dense"]["num_hidden_layers"]

        # #### Check whether fusion model is compatible with this classification head
        fusion_model_output_shape = self.fusion_model.get_shape_output_features()
        if len(fusion_model_output_shape) > 1:
            raise TypeError("Fusion Model is not compatible, as the Dense_ClassificationHead only accepts a flatten input!")

        # #### define layers
        self.hidden_layers = nn.ModuleList()

        act_fct_type = classification_head_config_dict["dense"]["act_fct"]
        num_input_neurons = int(fusion_model_output_shape[0])

        # add configured amount of hidden layers to dense classification head
        for layer_index in range(self.num_hidden_layers):
            hidden_layer = nn.ModuleDict()

            # get number of neurons for the layer from config and add it to ModuleDict
            try:
                neurons_in_layer = classification_head_config_dict["dense"]["neurons_at_layer_index"][layer_index]
            except IndexError:
                raise TypeError('Configuration for Dense_ClassificationHead does not contain enough entries for the parameter "neurons_at_layer_index"!')
            hidden_layer["fc"] = nn.Linear(num_input_neurons, neurons_in_layer)

            # add activation function based on config to ModuleDict
            if act_fct_type == "relu":
                hidden_layer["act_fct"] = nn.ReLU()
            elif act_fct_type == "gelu":
                hidden_layer["act_fct"] = nn.GELU()
            else:
                raise TypeError(
                    f"Activation function of Dense_ClassificationHead(Layer) is {act_fct_type} which is not a valid value!")

            # add dropout to hidden_layer ModuleDict if configured
            if layer_index in classification_head_config_dict["dense"]["dropout_at_layer_index"]:
                hidden_layer["dropout"] = nn.Dropout1d(classification_head_config_dict["dropout_rate"])

            # store number of neurons of this layer as number input neurons for next layer
            num_input_neurons = neurons_in_layer

            self.hidden_layers.append(hidden_layer)

        # add final linear layer
        self.fc_final = nn.Linear(num_input_neurons, num_classes)

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
        # hidden layers
        for layer_index in range(self.num_hidden_layers):
            x = self.hidden_layers[layer_index]["fc"](x)
            x = self.hidden_layers[layer_index]["act_fct"](x)

            # dropout can only be applied if present at layer_index
            if "dropout" in self.hidden_layers[layer_index].keys():
                x = self.hidden_layers[layer_index]["dropout"](x)

        # final classification layer
        x = self.fc_final(x)

        return x

# class Template_ClassificationHead(MultiModalBaseClass):
#     """
#         Template classification head as baseline for new classification heads.
#     """

#     def __init__(self, num_classes, sensors, classification_head_config_dict, fusion_model):
#         """
#             Init method of Template_ClassificationHead() class.

#             Parameters:
#                 - num_classes (int): Number of output neurons/ classes.
#                 - sensors (list): List of all sensors which shall be used
#                 - classification_head_config_dict (dict): Dict containing the classification head specific configuration parameters
#                 - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass
#         """
#         # #### call init method of superclass 
#         super().__init__(num_classes, sensors, classification_head_config_dict, fusion_model)

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