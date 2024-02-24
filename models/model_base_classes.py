import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelBaseClass(nn.Module):
    """
        Base Class for all ModalityNets and FusionModels for defining common methods only once.
    """

    def __init__(self):
        """
            Init method of BaseModelClass() class.
        """
        # __init__() method of superclass (nn.Module) must be called to be able to define layers.
        super().__init__()

    def calculate_features_from_sample_batch(self, sample_batch):
        """
            Method to calculate and store the number of flatten output features the shape of the output in members.

            Parameters:
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        self.num_flat_output_features = self.__get_num_flat_output_features(
            sample_batch)
        self.shape_output_features = self.__get_shape_output_features(
            sample_batch)

    def num_flat_features(self, x):
        """
            Method to calculate the number of features for a representation vector x.

            Parameters:
                - x (torch.Tensor): Representation vector

            Returns:
                - num_features (int): Number of features of the representation vector
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return int(num_features)

    def get_num_flat_output_features(self):
        """
            Public method to get the value of self.num_flat_output_features

            Returns:
                - (int): Number of flatten output of the model
        """
        return self.num_flat_output_features

    def __get_num_flat_output_features(self, sample_batch):
        """
            Private method to calculate the number of flatten output of the model based on a sample_batch

            Parameters:
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for

            Returns:
                - (int): Number of flatten output of the model
        """
        x = self.forward(sample_batch)
        return int(self.num_flat_features(x))

    def get_shape_output_features(self):
        """
            Public method to get the value of self.shape_output_features

            Returns:
                - (torch.Tensor): Shape of the output of the model
        """
        return self.shape_output_features

    def __get_shape_output_features(self, sample_batch):
        """
            Private method to get the shape of the models output based on a sample_batch

            Parameters:
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for

            Returns:
                - (torch.Tensor): Shape of the output of the model
        """
        x = self.forward(sample_batch)
        return x.size()[1:]


class ModalityNetBaseClass(ModelBaseClass):
    """
        Base class for all modality nets.
    """

    def __init__(self, sensor, modality_net_config_dict):
        """
            Init method of ModalityNetBaseClass() class.

            Parameters:
                - sensor (str): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
        """
        super().__init__()

        self.sensor = sensor
        self.modality_net_config_dict = modality_net_config_dict

    def forward(self, data_dict):
        """
            Default implementation of forward() method shall not be callable and thus will raise an exception.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Raises:
                - RuntimeError: Error whenever this function is called = must be overwritten by subclass
        """
        raise RuntimeError(
            "Method forward() of ModalityNetBaseClass shall not be called which means:\n\t\t"
            "1. Instance of ModalityNetBaseClass can't be used for training, ...\n\t\t"
            "2. forward() method must be overwritten by subclass")

    def confirm_input_is_an_image(self, sample_batch):
        """
            Method to check data from sample batch for compatibility (images expected). Method shall be called by subclass.

            Parameters:
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for

            Raises:
                TypeError: Raised if input data from sample batch is not a batch of images
        """
        if len(sample_batch[self.sensor].shape) != 4:
            raise TypeError(
                f"This modality net is not compatible with input of sensor {self.sensor}, as images with additional batch dimension are expected (batch, channels, x, y) as input!")

    def confirm_input_is_timeseries_data(self, sample_batch):
        """
            Method to check data from sample batch for compatibility (time series data expected). Method shall be called by subclass.

            Parameters:
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for

            Raises:
                TypeError: Raised if input data from sample batch is not a batch of timeseries data
        """
        if len(sample_batch[self.sensor].shape) != 3:
            raise TypeError(
                f"This modality net is not compatible with input of sensor {self.sensor}, as timeseries data with additional batch dimension is expected (batch, channels, data points) as input!")


class FusionModelBaseClass(ModelBaseClass):
    """
        Base class for all fusion models.
    """

    def __init__(self, sensors, fusion_model_config_dict, modality_nets: nn.ModuleDict):
        """
            Init method of FusionModelBaseClass() class.

            Parameters:
                - sensors (list): List of all sensors which shall be used
                - fusion_model_config_dict (dict): Dict containing the fusion model specific configuration parameters
                - modality_nets (nn.ModuleDict): ModuleDict containing all modality nets which must be a subclass of ModalityNetBaseClass
        """
        super().__init__()

        self.sensors = sensors
        self.fusion_model_config_dict = fusion_model_config_dict
        self.modality_nets = modality_nets

        self.check_compatibility()

    def check_compatibility(self):
        """
            Method checks, whether the input is compatible.

            Raises:
                - TypeError: self.modality_nets is not of type nn.ModuleDict
                - TypeError: At least one provided modality net is not a direct subclass of ModalityNetBaseClass class
        """

        # check whether the modality nets are provided as nn.ModuleDict
        if type(self.modality_nets) != nn.ModuleDict:
            raise TypeError(
                "Modality nets are not provided in a nn.ModuleDict!")

        # check type of each each modality net
        modality_nets_are_compatible = True
        for _, modality_net in self.modality_nets.items():
            # will become false, if modality_net is not a driect subclass of ModalityNetBaseClass
            modality_nets_are_compatible = modality_nets_are_compatible and (
                ModalityNetBaseClass == type(modality_net).__mro__[1])

        if not modality_nets_are_compatible:
            raise TypeError(
                "At least one provided modality net is not a direct subclass of ModalityNet class!")

    def forward(self, data_dict):
        """
            Default implementation of forward() method shall not be callable and thus will raise an exception.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Raises:
                - RuntimeError: Error whenever this function is called = must be overwritten by subclass
        """
        raise RuntimeError(
            "Method forward() of FusionModelBaseClass shall not be called which means:\n\t\t"
            "1. Instance of FusionModelBaseClass can't be used for training, ...\n\t\t"
            "2. forward() method must be overwritten by subclass")


class MultiModalBaseClass(nn.Module):
    """
        Base class for all multi-modal models.
    """

    def __init__(self, num_classes, sensors, classification_head_config_dict, fusion_model: FusionModelBaseClass):
        """
            Init method of MultiModalBaseClass() class.

            Parameters:
                - num_classes (int): Number of output neurons/ classes.
                - sensors (list): List of all sensors which shall be used
                - classification_head_config_dict (dict): Dict containing the classification head specific configuration parameters
                - fusion_model (FusionModelBaseClass): Fusion model instance to be used which must be a subclass of FusionModelBaseClass
        """
        super().__init__()

        self.num_classes = num_classes
        self.sensors = sensors
        self.classification_head_config_dict = classification_head_config_dict
        self.fusion_model = fusion_model

        self.check_compatibility()

    def check_compatibility(self):
        """
            Method checks, whether the input is compatible.

            Raises:
                - TypeError: The provided fusion model is not a direct subclass of FusionModelBaseClass class!
        """
        if not (FusionModelBaseClass == type(self.fusion_model).__mro__[1]):
            raise TypeError(
                "The provided fusion model is not a direct subclass of FusionModel class!")

    def forward(self, data_dict):
        """
            Default implementation of forward() method shall not be callable and thus will raise an exception.

            Parameters:
                - data_dict (torch.Tensor): Batch from dataset to process

            Raises:
                - RuntimeError: Error whenever this function is called = must be overwritten by subclass
        """
        raise RuntimeError(
            "Method forward() of MultiModalBaseClass shall not be called which means:\n\t\t"
            "1. Instance of MultiModalBaseClass can't be used for training, ...\n\t\t"
            "2. forward() method must be overwritten by subclass")
