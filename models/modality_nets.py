import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import ModalityNetBaseClass
from models.tokenEmbeddings import flatten_patches, create_patch_sequence_for_image


class ImagePatchTokenization_ModalityNet(ModalityNetBaseClass):
    """
        Template modality net as baseline for new modality nets.
    """

    def __init__(self, sensor, modality_net_config_dict, sample_batch):
        """
            Init method for the ImagePatchTokenization_ModalityNet.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, modality_net_config_dict)

        # #### check whether input is compatible
        self.confirm_input_is_an_image(sample_batch)

        # #### get shape of flatten patches for ViT strategy
        patch_size = modality_net_config_dict["PatchTokenization"]["patch_size"]
        patch_dim = flatten_patches(create_patch_sequence_for_image(sample_batch[self.sensor][0], patch_size)).shape[1]

        # #### define layers
        if modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "vit":
            # linear layer (= without bias) for linear projection of patches to embedding dim according to ViT (https://arxiv.org/pdf/2010.11929.pdf)
            self.linear = nn.Linear(patch_dim, modality_net_config_dict["PatchTokenization"]["embed_dim"], bias=False)

        elif modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "metaTransformer":
            pass

        else:
            raise TypeError(
                f"Image tokenization strategy for ImagePatchTokenization_ModalityNet Model is {modality_net_config_dict['PatchTokenization']['image_tokenization_strategy']} which is not a valid value!")

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
        # ## transform images to patches first for complete batch
        x = data_dict[self.sensor]
        batch_size = x.shape[0]

        patches_first_batch = create_patch_sequence_for_image(x[0], self.modality_net_config_dict["PatchTokenization"]["patch_size"])
        num_patches = patches_first_batch.shape[0]

        patches_for_all_batches = torch.zeros(batch_size, num_patches, patches_first_batch.shape[1], patches_first_batch.shape[2], patches_first_batch.shape[3])
        patches_for_all_batches[0] = patches_first_batch

        for i in range(1, batch_size):
            patches_for_all_batches[i] =  create_patch_sequence_for_image(x[i], self.modality_net_config_dict["PatchTokenization"]["patch_size"])

        # ## rest of the image tokenization is different for configurable strategies
        if self.modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "vit":
            size_flatten_patch = patches_for_all_batches[0].shape[1] * patches_for_all_batches[0].shape[2] * patches_for_all_batches[0].shape[3]

            x = torch.zeros(batch_size, num_patches, size_flatten_patch, dtype=torch.float32)

            for i in range(batch_size):
                x[i] = flatten_patches(patches_for_all_batches[i])

            x = self.linear(x)

        elif self.modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "metaTransformer":
            pass

        return x

class LeNet2dLike_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based on the LeNet-5 architecture. 
        Paper introducing the architecture: https://ieeexplore.ieee.org/document/726791
    """

    def __init__(self, sensor, modality_net_config_dict, sample_batch):
        """
            Init method of LeNet2dLike_ModalityNet() class.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, modality_net_config_dict)

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

    def __init__(self, sensor, modality_net_config_dict, sample_batch):
        """
            Init method for the LeNet1dLike_ModalityNet.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, modality_net_config_dict)

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


class VggLike_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based on the VGG architecture.
        Paper introducing the architecture: https://arxiv.org/abs/1409.1556
    """

    def __init__(self, sensor, modality_net_config_dict, sample_batch):
        """
            Init method for the Template_ModalityNet.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, modality_net_config_dict)

        # #### check whether input is compatible
        self.confirm_input_is_an_image(sample_batch)

        # #### define layers
        kernel_size = 3
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(16, 64, kernel_size)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

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
        x = data_dict[self.sensor]

        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))

        x = F.relu(self.conv3_1(x))
        x = self.pool3(F.relu(self.conv3_2(x)))

        x = self.gap(x)
        x = self.flatten(x)
        return x

# class Template_ModalityNet(ModalityNetBaseClass):
#     """
#         Template modality net as baseline for new modality nets.
#     """

#     def __init__(self, sensor, modality_net_config_dict, sample_batch):
#         """
#             Init method for the Template_ModalityNet.

#             Parameters:
#                 - sensor (string): Name of the sensor for which this modality net is used
#                 - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
#                 - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
#         """
#         # #### call init method of superclass
#         super().__init__(sensor, modality_net_config_dict)

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
