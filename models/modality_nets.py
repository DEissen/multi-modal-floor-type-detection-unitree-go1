import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base_classes import ModalityNetBaseClass
from models.tokenEmbeddings import flatten_patches, create_patch_sequence_for_image


class ImagePatchTokenization_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based for patch tokenization of images for creating a sequence of 1D embedding vectors for transformers.
        Supports tokenization strategy from ViT (https://arxiv.org/pdf/2010.11929.pdf) and Meta Transformer (https://arxiv.org/pdf/2307.10802.pdf).
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

        # #### define layers
        if modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "vit":
            # get shape of flatten patches for ViT strategy
            patch_size = modality_net_config_dict["PatchTokenization"]["patch_size"]
            patch_dim = flatten_patches(create_patch_sequence_for_image(
                sample_batch[self.sensor][0], patch_size), self.device).shape[1]

            # linear layer (= without bias) for linear projection of patches to embedding dim according to ViT (https://arxiv.org/pdf/2010.11929.pdf)
            self.linear = nn.Linear(
                patch_dim, modality_net_config_dict["PatchTokenization"]["embed_dim"], bias=False)

        elif modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "metaTransformer":
            # implementation based on https://github.com/invictus717/MetaTransformer/blob/master/Data2Seq/Image.py
            patch_size = modality_net_config_dict["PatchTokenization"]["patch_size"]
            patch_size = (patch_size, patch_size)
            self.proj_conv = nn.Conv2d(
                3, modality_net_config_dict["PatchTokenization"]["embed_dim"], kernel_size=patch_size, stride=patch_size)

        elif modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "LeNetLike":
            # using LeNet2dLike_ModalityNet() class with additional Dense Layer to get embedding dimension
            self.le_net_like = LeNet2dLike_ModalityNet(
                sensor, modality_net_config_dict, sample_batch)

            # representations after LeNet2dLike_ModalityNet will have shape [new_c, new_h, new_w]
            # This will be flatten to a tensor of shape [new_c, new_h*new_w] which will be input for the dense layer
            # => Input dimension for dense layer is new_h*new_w and is calculated below
            input_dim = self.le_net_like.get_shape_output_features()[1] * \
                self.le_net_like.get_shape_output_features()[2]

            self.linear = nn.Linear(
                input_dim, modality_net_config_dict["PatchTokenization"]["embed_dim"], bias=False)

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
        # ## common part
        x = data_dict[self.sensor]

        # ## rest of the image tokenization is different for configurable strategies
        if self.modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "vit":
            # transform images to patches first for complete batch
            batch_size = x.shape[0]

            patches_first_batch = create_patch_sequence_for_image(
                x[0], self.modality_net_config_dict["PatchTokenization"]["patch_size"])
            num_patches = patches_first_batch.shape[0]

            patches_for_all_batches = torch.zeros(
                batch_size, num_patches, patches_first_batch.shape[1], patches_first_batch.shape[2], patches_first_batch.shape[3], device=self.device)
            patches_for_all_batches[0] = patches_first_batch

            for i in range(1, batch_size):
                patches_for_all_batches[i] = create_patch_sequence_for_image(
                    x[i], self.modality_net_config_dict["PatchTokenization"]["patch_size"])

            # flatten patches and project with linear layer to embedding dimension
            size_flatten_patch = patches_for_all_batches[0].shape[1] * \
                patches_for_all_batches[0].shape[2] * \
                patches_for_all_batches[0].shape[3]

            x = torch.zeros(batch_size, num_patches,
                            size_flatten_patch, dtype=torch.float32, device=self.device)

            for i in range(batch_size):
                x[i] = flatten_patches(patches_for_all_batches[i], device=self.device)

            x = self.linear(x)

        elif self.modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "metaTransformer":
            # implementation based on https://github.com/invictus717/MetaTransformer/blob/master/Data2Seq/Image.py
            # projection layer (Conv2D) will lead to projections in shape [batch_size, new_c, new_h, new_w]
            # flatten leads to projections in shape [batch_size, new_c, new_h*new_w]
            # transpose leads to projections in shape [batch_size, new_h*new_w, new_c]
            # => Thus new_c determined by Conv2D layer is embedding dimension and number of tokens is determined by input image size and patch size (used as kernel size and stride)
            x = self.proj_conv(x).flatten(2).transpose(1, 2)

        elif self.modality_net_config_dict["PatchTokenization"]["image_tokenization_strategy"] == "LeNetLike":
            # flatten features from LeNetLike results in shape in shape [batch_size, new_c, new_h*new_w]
            x = self.le_net_like(data_dict).flatten(2)
            # linear layer maps to needed shape [batch_size, new_c, embed_dim]
            x = self.linear(x)

        return x


class TimeseriesTokenization_ModalityNet(ModalityNetBaseClass):
    """
        Modality net based for creating a sequence of 1D embedding vectors for transformers.
        Using LeNet1dLike_ModalityNet() as feature extractor or patch tokenization similar to Meta Transformers (https://arxiv.org/pdf/2307.10802.pdf) image patch tokenization but with 1D Conv.
    """

    def __init__(self, sensor, modality_net_config_dict, sample_batch):
        """
            Init method for the TimeseriesPatchTokenization_ModalityNet.

            Parameters:
                - sensor (string): Name of the sensor for which this modality net is used
                - modality_net_config_dict (dict): Dict containing the modality net specific configuration parameters
                - sample_batch (torch.Tensor): One batch from the dataset the model shall be used for
        """
        # #### call init method of superclass
        super().__init__(sensor, modality_net_config_dict)

        # #### check whether input is compatible
        self.confirm_input_is_timeseries_data(sample_batch)

        # #### define layers
        if modality_net_config_dict["PatchTokenization"]["timeseries_tokenization_strategy"] == "LeNetLike":
            # use LeNet1dLike_ModalityNet with additional linear layer to project representation to embedding dimension
            self.le_net_like = LeNet1dLike_ModalityNet(
                sensor, modality_net_config_dict, sample_batch)
            patch_dim = self.le_net_like.get_shape_output_features()[1]
            self.linear = nn.Linear(
                patch_dim, modality_net_config_dict["PatchTokenization"]["embed_dim"], bias=False)

        elif modality_net_config_dict["PatchTokenization"]["timeseries_tokenization_strategy"] == "metaTransformer":
            # implementation based on https://github.com/invictus717/MetaTransformer/blob/master/Data2Seq/Image.py but with 1D Conv
            kernel_size = modality_net_config_dict["PatchTokenization"]["kernel_size"]
            in_c = sample_batch[sensor].shape[1]
            self.proj_conv = nn.Conv1d(in_c, modality_net_config_dict["PatchTokenization"]["embed_dim"], kernel_size=kernel_size,
                                       stride=kernel_size)
        else:
            raise TypeError(
                f"Image tokenization strategy for TimeseriesPatchTokenization_ModalityNet Model is {modality_net_config_dict['PatchTokenization']['timeseries_tokenization_strategy']} which is not a valid value!")

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
        # ## rest of the image tokenization is different for configurable strategies
        if self.modality_net_config_dict["PatchTokenization"]["timeseries_tokenization_strategy"] == "LeNetLike":
            x = self.le_net_like(data_dict)

            x = self.linear(x)

        elif self.modality_net_config_dict["PatchTokenization"]["timeseries_tokenization_strategy"] == "metaTransformer":
            x = data_dict[self.sensor]

            # implementation based on https://github.com/invictus717/MetaTransformer/blob/master/Data2Seq/Image.py but with 1D Conv
            # projection layer (Conv1D) will lead to projections in shape [batch_size, new_c, new_w]
            # transpose leads to projections in shape [batch_size, new_w, new_c]
            # => Thus new_c determined by Conv1D layer is embedding dimension and number of tokens is determined by input window size and kernel size (used as kernel size and stride)
            x = self.proj_conv(x).transpose(1, 2)

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
