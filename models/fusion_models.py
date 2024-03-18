import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

from models.model_base_classes import FusionModelBaseClass
from models.transformers import CrossModalTransformer


class Concatenate_FusionModel(FusionModelBaseClass):
    """
        Class for simple feature fusion with concatenation of flatten outputs.
    """

    def __init__(self, sensors, fusion_model_config_dict, modality_nets, sample_batch):
        """
            Init method of ConcatenateForFusion() class.

            Parameters:
                - sensors (list): List containing all sensors which are present in the datasets
                - fusion_model_config_dict (dict): Dict containing the fusion model specific configuration parameters
                - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensors, fusion_model_config_dict, modality_nets)

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
        # ### get the result of each modality net
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


class CrossModalTransformer_FusionModel(FusionModelBaseClass):
    """
         Fusion model using cross-modal Transformers.
    """

    def __init__(self, sensors, fusion_model_config_dict, modality_nets, sample_batch):
        """
            Init method of Template_FusionModel() class.

            Parameters:
                - sensors (list): List containing all sensors which are present in the datasets
                - fusion_model_config_dict (dict): Dict containing the fusion model specific configuration parameters
                - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensors, fusion_model_config_dict, modality_nets)

        # #### check whether at least two sensors are provided
        if len(sensors) < 2:
            raise TypeError(f"CrossModalTransformer_FusionModel can only be used in multi-modal context (at least two sensors must be provided)!") 

        # #### check whether modality_nets are compatible
        for sensor in sensors:
            modality_net_embed_dim = modality_nets[sensor].get_shape_output_features()[1]
            
            if fusion_model_config_dict['CrossModalTransformer']['embed_dim'] != modality_net_embed_dim:
                raise TypeError(f"Embedding dimension does not fit for modality net of {sensor}!") 

        # #### get combinations for cross-modal Transformers based on configured method
        if fusion_model_config_dict["CrossModalTransformer"]["fusion_strategy"] == "highMMT":
            self.fusion_combinations = self.__get_highMMT_combinations(sensors)
        elif fusion_model_config_dict["CrossModalTransformer"]["fusion_strategy"] == "mult":
            self.fusion_combinations = self.__get_MulT_combinations(sensors)
        else:
            raise TypeError(
                f"Fusion strategy for CrossModalTransformer_Fusion Model is {fusion_model_config_dict['CrossModalTransformer']['fusion_strategy']} which is not a valid value!")

        # #### define layers
        # CrossModalTransformers are organized in multimodal streams, where in each stream the output of multiple CrossModalTransformers is concatenated based on the fusion strategy
        self.multimodal_streams = nn.ModuleList([])

        for multimodal_stream_combinations in self.fusion_combinations:
            # To identify each transformer model within a multimodal stream, each stream is organized as a ModuleDict
            multimodal_stream = nn.ModuleDict()

            for combination in multimodal_stream_combinations:
                # create Key for the ModuleDict in notation from MA3606 based on MulT (https://arxiv.org/pdf/1906.00295.pdf)
                target_mod = combination[0]
                source_mod = combination[1]
                key = source_mod + "=>" + target_mod

                # create model for current combination
                multimodal_stream[key] = CrossModalTransformer(
                    target_mod, source_mod, fusion_model_config_dict["CrossModalTransformer"])

            self.multimodal_streams.append(multimodal_stream)

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
        # ### get the result of each modality net
        modality_net_outputs = {}
        for sensor in self.sensors:
            # append output/ extracted features for sensor to the concatenation list
            modality_net_outputs[sensor] = self.modality_nets[sensor](
                data_dict)

        # ### fusion model part
        multimodal_stream_outputs = []

        # iterate over all multimodal streams
        for multimodal_stream in self.multimodal_streams:
            # iterate over all keys in the stream and create a list of all outputs
            multimodal_stream_output = []
            multimodal_stream_keys = multimodal_stream.keys()
            for key in multimodal_stream_keys:
                source_sensor, target_sensor = key.split("=>")

                x = multimodal_stream[key](
                    modality_net_outputs[target_sensor], modality_net_outputs[source_sensor], target_sensor, source_sensor)
                
                # flatten the result (important if whole sequence is used with different sequence lengths)
                x = x.view(x.size(0), -1)

                multimodal_stream_output.append(x)

            # append concatenated results (concatenated in embed_dim as suggested by MulT (https://arxiv.org/pdf/1906.00295.pdf)) to multimodal_stream_outputs
            multimodal_stream_outputs.append(
                torch.cat(multimodal_stream_output, dim=1))
        
        # concatenate results of all multi-modal streams
        output = torch.cat(multimodal_stream_outputs, dim=1)

        # # optionally use Batch Normalization on results as suggested by HighMMT (https://arxiv.org/pdf/2203.01311.pdf)
        # NOTE: Not supported yet, as BN must be initialized with number of parameters provided which is not easily possible!
        # if self.fusion_model_config_dict["CrossModalTransformer"]["norm_output"]:
        #     output = output

        return output

    def __get_highMMT_combinations(self, sensors):
        """
            Private method to get list of sensor combinations for the cross-modal Transformers according to strategy from HighMMT (https://arxiv.org/pdf/2203.01311.pdf).
            Strategy results in one multi-modal stream for each unique sensor combination in both directions.

            Parameters:
                - sensors (list): List of all sensors which shall be considered for this dataset

            Returns:
                - highMMT_combinations_list (list of tuples): List of sensor combination as tuples in format (target_mod, source_mod)
        """
        highMMT_combinations_list = []

        for combination in list(combinations(sensors, 2)):
            highMMT_combinations_list.append(
                [combination, (combination[1], combination[0])])

        return highMMT_combinations_list

    def __get_MulT_combinations(self, sensors):
        """
            Private method to get list of sensor combinations for the cross-modal Transformers according to strategy from MulT (https://arxiv.org/pdf/1906.00295.pdf).
            Strategy results in one multi-modal stream for each target modality containing all possible combinations with the same target modality.

            Parameters:
                - sensors (list): List of all sensors which shall be considered for this dataset

            Returns:
                - highMMT_combinations_list (list of tuples): List of sensor combination as tuples in format (target_mod, source_mod)
        """
        MulT_combinations_list = []

        for target_mod in sensors:
            target_modality_list = []
            for source_mod in sensors:
                if target_mod != source_mod:
                    target_modality_list.append((target_mod, source_mod))

            MulT_combinations_list.append(target_modality_list)

        return MulT_combinations_list

# class Template_FusionModel(FusionModelBaseClass):
#     """
#          Template fusion model as baseline for new fusion models.
#     """

#     def __init__(self, sensors, fusion_model_config_dict, modality_nets, sample_batch):
#         """
#             Init method of Template_FusionModel() class.

#             Parameters:
#                 - sensors (list): List containing all sensors which are present in the datasets
#                 - fusion_model_config_dict (dict): Dict containing the fusion model specific configuration parameters
#                 - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
#                 - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
#         """
#         # #### call init method of superclass
#         super().__init__(sensors, fusion_model_config_dict, modality_nets)

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
