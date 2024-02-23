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
    from models.modality_nets import LeNet2dLike_ModalityNet, LeNet1dLike_ModalityNet


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

    # define modality nets
    modality_nets = torch.nn.ModuleDict()
    for sensor in sensors:
        if "Cam" in sensor:
            modality_nets[sensor] = LeNet2dLike_ModalityNet(
                sensor, model_config_dict, sample_batch)
        else:
            modality_nets[sensor] = LeNet1dLike_ModalityNet(
                sensor, model_config_dict, sample_batch)

    # define fusion model
    fusion_model = Concatenate_FusionModel(
        sensors, model_config_dict, modality_nets, sample_batch)

    # define final model
    final_model = Dense_ClassificationHead(
        num_classes, sensors, model_config_dict, fusion_model)

    return final_model
