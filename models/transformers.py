import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTransformerLayer(nn.Module):
    """
         Cross-modal Transformer based on description in MulT (https://arxiv.org/pdf/1906.00295.pdf) with optional latent self-attention 
         as described by HighMMT (https://arxiv.org/pdf/2203.01311.pdf).
    """

    def __init__(self, target_sensor: str, source_sensor: str, config_dict: dict):
        """
            Init method of CrossModalTransformerLayer() class.
            Cross-modal transformer will use data of target_sensor as Query and data of source_sensor as Key and Value
            which is denoted as source_sensor => target_sensor.
            Data of target_sensor and source_sensor must have the same shape ``[batch_size, seq_len, embed_dim]``.

            Parameters:
                - target_sensor (str): Name of the target sensor used as Query in cross-modal attention
                - source_sensor (str): Name of the source sensor used as Key and Value in cross-modal attention
                - config_dict (dict): Dict containing the CrossModalTransformer(Layer) specific configuration parameters
        """
        # #### call init method of superclass
        super().__init__()

        # #### store relevant parameters as members
        self.target_sensor = target_sensor
        self.source_sensor = source_sensor
        self.config_dict = config_dict

        # #### define layers
        # ## Sublayer with cross attention
        self.pre_cross_LN_for_target_mod = nn.LayerNorm(config_dict["embed_dim"])
        self.pre_cross_LN_for_source_mod = nn.LayerNorm(config_dict["embed_dim"])
        # Note:
        #   - batch_first is set to True as data is expected in shape [batch_size, seq_len, embed_dim]
        #   - bias is set to False to fit implementation from "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf) where no bias is mentioned
        self.cross_attention = nn.MultiheadAttention(
            config_dict["embed_dim"], config_dict["cross_num_heads"], config_dict["cross_attn_dropout"], bias=False, batch_first=True)

        # ## Optional latent self-attention sublayer
        if config_dict["use_latent_self_attn"]:
            self.pre_latent_LN = nn.LayerNorm(config_dict["embed_dim"])

            self.latent_self_attention = nn.MultiheadAttention(
                config_dict["embed_dim"], config_dict["latent_num_heads"], config_dict["latent_attn_dropout"], bias=False, batch_first=True)

        # ## Sublayer with FFN
        # Note:
        #   - input and output dim is embed_dim and inner-layer dim is 4*embed_dim as in "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
        self.pre_FFN_LN = nn.LayerNorm(config_dict["embed_dim"])
        self.fc1 = nn.Linear(
            config_dict["embed_dim"], 4*config_dict["embed_dim"])
        self.fc2 = nn.Linear(
            4*config_dict["embed_dim"], config_dict["embed_dim"])
        # choose activation function based on config
        if config_dict["act_fct"] == "relu":
            self.act_fct = nn.ReLU()
        elif config_dict["act_fct"] == "gelu":
            self.act_fct = nn.GELU()
        else:
            raise TypeError(f"Activation function of CrossModalTransformer(Layer) is {config_dict['act_fct']} which is not a valid value!")
        

    def forward(self, target_sensor_data: torch.Tensor, source_sensor_data: torch.Tensor, target_sensor: str, source_sensor: str):
        """
            Implementation of forward() method to process data.
            Data of target_sensor and source_sensor must have the same shape ``[batch_size, seq_len, embed_dim]``.

            Parameters:
                - target_sensor_data (torch.Tensor): Data of the target sensor used as Query in cross-modal attention
                - source_sensor_data (torch.Tensor): Data of the source sensor used as Key and Value in cross-modal attention
                - target_sensor (str): Name of the sensor which provided target_sensor_data
                - source_sensor (str): Name of the sensor which provided source_sensor_data

            Returns:
                - x (torch.Tensor): Representation vector after processing the data
        """
        # ## check wether provided target_sensor and source_sensor fit to expectation
        if self.target_sensor != target_sensor or self.source_sensor != source_sensor:
            raise TypeError(f"CrossModalTransformer(Layer) for {self.source_sensor+'=>'+self.target_sensor} received wrong data for: {source_sensor+'=>'+target_sensor}!")

        # ## Sublayer with cross attention
        residual = target_sensor_data

        x_q = self.pre_cross_LN_for_target_mod(target_sensor_data)
        x_k_v = self.pre_cross_LN_for_source_mod(source_sensor_data)

        # weights from attention layer are not needed (which is enabled by default)
        x, _ = self.cross_attention(x_q, x_k_v, x_k_v, need_weights=False)

        x = x + residual

        # ## Optional latent self-attention sublayer
        if self.config_dict["use_latent_self_attn"]:
            residual = x

            x = self.pre_latent_LN(x)

            # weights from attention layer are not needed (which is enabled by default)
            x, _ = self.latent_self_attention(x, x, x, need_weights=False)

            x = x + residual
            
        # ## Sublayer with FFN
        residual = x

        x = self.pre_FFN_LN(x)

        # FFN path
        x = self.act_fct(self.fc1(x))
        x = self.fc2(x)
        # apply dropout to output of sublayer, before it is added to it's input as in "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
        x = F.dropout(x, self.config_dict["ffn_dropout"])

        x = x + residual 

        return x
