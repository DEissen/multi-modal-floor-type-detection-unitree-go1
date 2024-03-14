import torch
from torch import nn, Tensor
import math


class VanillaPositionalEncoding(nn.Module):
    """
        Fixed Positonal Encoding (PE) from Vanilla Transformer. Based on implementation from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        - Same Fixed PE's are used for every batch
        - Input sequence must be provided with shape ``[batch_size, seq_len, embedding_dim]`` 
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 500):
        """
            Init method for VanillaPositionalEncoding() class.

            Parameters:
                - d_model (int): Dimension of each Token for which PE's shall be created
                - dropout (float, optional): Optional dropout applied to returned sequence. Defaults to 0.0.
                - max_len (int, optional): Max sequence length for which PE's shall be provided. Defaults to 500.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # calculate PE's once for max_len and store them in buffer self.pe
        # create position values for sequence starting at 0
        position = torch.arange(max_len).unsqueeze(1)
        # get values of div term inside cos/ sin function to multiply with position value
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # initialize values for PE
        # calculate values for odd positions with sin function
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # calculate values for even positions with cos function
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
            Forward function to add PE's to a Batch of an token sequence with shape ``[batch_size, seq_len, embedding_dim]``.

            Parameters:
                - x (torch.Tensor): Batch of input data to add PE's to

            Returns:
                - x (torch.Tensor): Batch of input data with PE's added to it
        """
        input_seq_len = x.size(1)  # get sequence length from input
        # add PE's fitting to sequence length of the input
        x = x + self.pe[:, :input_seq_len, :]
        x = self.dropout(x)
        return x
