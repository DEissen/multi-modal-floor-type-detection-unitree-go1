import torch
from torch.utils.data import DataLoader

# custom imports
if __name__ == "__main__":
    from classification_heads import Dense_ClassificationHead
    from fusion_models import Concatenate_FusionModel
    from modality_nets import LeNet2dLike_ModalityNet
else:
    from models.classification_heads import Dense_ClassificationHead
    from models.fusion_models import Concatenate_FusionModel
    from models.modality_nets import LeNet2dLike_ModalityNet, LeNet1dLike_ModalityNet, VggLike_ModalityNet


def model_builder(num_classes, sensors, dataset, model_config_dict):
    """
        Create uni-model or multi-modal model based on used sensors, fitting for the dataset according to the configuration from config_dict.

        Parameters:
            - num_classes (int): Number of output neurons/ classes.
            - sensors (list): List of all sensors which shall be considered for this dataset
            - dataset (torch.utils.data.Dataset): Dataset for which the model shall be used
            - model_config_dict (dict): Dict containing the configuration for the testing

        Returns:
            - final_model: The fitting uni-modal or multi-modal model
    """
    # get a sample batch for model creation (to determine torch.Tensor shapes)
    sample_loader = DataLoader(dataset, 8)
    _, (sample_batch, _) = next(enumerate(sample_loader))

    # get modality nets
    modality_nets = torch.nn.ModuleDict()
    for sensor in sensors:
        modality_nets[sensor] = get_modality_net_based_on_config(
            sensor, model_config_dict["modality_net"], sample_batch)

    # get fusion model
    fusion_model = get_fusion_model_based_on_config(
        sensors, model_config_dict["fusion_model"], modality_nets, sample_batch)

    # get final model
    final_model = get_final_model_based_on_config(
        num_classes, sensors, model_config_dict["classification_head"], fusion_model)

    return final_model


def get_modality_net_based_on_config(sensor, modality_net_config_dict, sample_batch):
    """
        Function to return modality net for sensor based on config in modality_net_config_dict.

        Parameters:
            - sensor (str): Name of the sensor from the dataset
            - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
            - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for

        Returns:
            - modality_net (subclass of ModalityNetBaseClass): Modality net for the sensor
    """
    # initialize modality net with None for later check whether modality net was created
    modality_net = None

    # select modality net for cameras
    if "Cam" in sensor:
        if modality_net_config_dict["type"] == "LeNetLike":
            modality_net = LeNet2dLike_ModalityNet(
                sensor, modality_net_config_dict, sample_batch)
        elif modality_net_config_dict["type"] == "VggLike":
            modality_net = VggLike_ModalityNet(
                sensor, modality_net_config_dict, sample_batch)
    # select modality net for timeseries data
    else:
        modality_net = LeNet1dLike_ModalityNet(
            sensor, modality_net_config_dict, sample_batch)

    # raise Exception in case modality net creation failed (due to missing/ wrong configuration)
    if modality_net == None:
        raise RuntimeError(
            f"Modality net selection for {sensor} was not successful. Please check the configuration!")

    return modality_net


def get_fusion_model_based_on_config(sensors, fusion_model_config_dict, modality_nets, sample_batch):
    """
        Function to return fusion model based on config in fusion_model_config_dict.

        Parameters:
                - sensors (list): List containing all sensors which are present in the datasets
                - fusion_model_config_dict (dict): Dict containing the fusion model specific configuration parameters
                - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        Returns:
            - fusion_model (subclass of FusionModelBaseClass): Fusion model
    """
    # initialize modality net with None for later check whether modality net was created
    fusion_model = None

    if fusion_model_config_dict["type"] == "concat":
        fusion_model = Concatenate_FusionModel(
            sensors, fusion_model_config_dict, modality_nets, sample_batch)

    # raise Exception in case modality net creation failed (due to missing/ wrong configuration)
    if fusion_model == None:
        raise RuntimeError(
            "Fusion model selection was not successful. Please check the configuration!")

    return fusion_model


def get_final_model_based_on_config(num_classes, sensors, classification_head_config_dict, fusion_model):
    """
        Function to return final model with classification head based on config in classification_head_config_dict.

        Parameters:
            - num_classes (int): Number of output neurons/ classes.
            - sensors (list): List of all sensors which shall be used
            - classification_head_config_dict (dict): Dict containing the classification head specific configuration parameters
            - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass

        Returns:
            - final_model (subclass of ModalityNetBaseClass): Final model with classification head
    """
    # initialize modality net with None for later check whether modality net was created
    final_model = None

    if classification_head_config_dict["type"] == "dense":
        final_model = Dense_ClassificationHead(
            num_classes, sensors, classification_head_config_dict, fusion_model)

    # raise Exception in case modality net creation failed (due to missing/ wrong configuration)
    if final_model == None:
        raise RuntimeError(
            "Final model creation was not successful. Please check the configuration!")

    return final_model
