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
    patches = img.data.unfold(0, 3, 3).unfold(
        1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # output contains one extra dim for each unfold operation => shape will be (unfold_c, unfold_h, unfold_w, c, h, w)
    # thus patches must be reshaped to a single sequence
    num_patches = patches.shape[0] * patches.shape[1] * patches.shape[2]
    patch_seq = patches.reshape(
        (num_patches, patches.shape[3], patches.shape[4], patches.shape[5]))

    return patch_seq


def flatten_patches(img_patches, device):
    """
        The function expects a sequence of image patches in shape [num_patches, c, h, w] which are flatten to a 1D sequence of shape [num_patches, c*h*w]

        Parameters:
            - img_patches (torch.Tensor): Sequence of image patches in shape [num_patches, c, h, w]

        Returns:
            - res (torch.Tensor): Sequence of flatten patches in shape [num_patches, c*h*w]
    """
    num_patches = img_patches.shape[0]
    size_flatten_patch = img_patches.shape[1] * \
        img_patches.shape[2] * img_patches.shape[3]

    res = torch.zeros(num_patches, size_flatten_patch,
                      dtype=torch.float32, device=device)
    for num_patch in range(num_patches):
        res[num_patch] = img_patches[num_patch].flatten()

    return res
