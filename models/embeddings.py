import torch
import torch.nn as nn
import torch.nn.functional as F

def create_patch_sequence_for_image(img: torch.Tensor, patch_size: int):
    """
        Create sequence of patches of size patch_size*patch_size from image provided as torch.Tensor.

        Parameters:
            - img (torch.Tensor): Image to create patches from
            - patch_size (int): Size of the patches

        Returns:
            - patch_seq (torch.Tensor): Sequence of patches of the image in shape [seq_len, c, h, w]
    """
    # image will be sliced in patches of size patch_size*patch_size by using function torch.Tensor.unfold(dimension, size, step)
    # unfold() must be applied three times, once for each channel
    patches = img.data.unfold(0, 3, 3).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # output contains one extra dim for each unfold operation => shape will be (unfold_c, unfold_h, unfold_w, c, h, w)
    # thus patches must be reshaped to a single sequence
    num_patches = patches.shape[0] * patches.shape[1] * patches.shape[2]
    patch_seq = patches.reshape(
        (num_patches, patches.shape[3], patches.shape[4], patches.shape[5]))

    return patch_seq