from typing import Optional, Tuple, Union
import torch 
from diffusers import UNet2DConditionModel
from geometry_encoder import GeometryEncoder


class FlowGenerator(torch.nn.Module):
    """
    FlowGenerator is a module that generates flow predictions based on given conditions and geometry.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        geometry_channels (int): Number of channels in the geometry input.
        down_block_types (Tuple[str, ...]): Types of down blocks to use in the UNet2DConditionModel.
        geometry_block_types (Tuple[str, ...]): Types of blocks to use in the GeometryEncoder.
        mid_block_type (str): Type of block to use in the UNet2DConditionModel for the mid block.
        up_block_types (Tuple[str, ...]): Types of up blocks to use in the UNet2DConditionModel.
        block_out_channels (Tuple[int, ...]): Number of output channels for each block in the UNet2DConditionModel.

    Attributes:
        condition_unet (UNet2DConditionModel): UNet2DConditionModel for condition encoding.
        geometry_encoder (GeometryEncoder): GeometryEncoder for geometry encoding.

    """

    def __init__(self, 
        in_channels: int = 3,
        out_channels: int = 3, 
        geometry_channels: int = 3,
        layers_per_block: int = 2,
        down_block_types: Tuple[str, ...] =("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        mid_block_type: str = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str, ...] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        geometry_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
    ):

        super().__init__()  # Initialize the parent class (torch.Module)

        # Initialize the GeometryEncoder 
        self.geometry_encoder = GeometryEncoder(
            in_channels=geometry_channels,
            layers_per_block=layers_per_block,
            down_block_types=geometry_block_types,
            block_out_channels=block_out_channels)

        # Initialize the UNet2DConditionModel 
        self.condition_unet = UNet2DConditionModel(
            in_channels=in_channels, 
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            cross_attention_dim=block_out_channels[-1],
            # encoder_hid_dim_type='image',
            # encoder_hid_dim = block_out_channels[-1],
        )
        

    def forward(self, sample, condition, time):
        """
        Forward pass of the FlowGenerator module.

        Args:
            sample: Input sample.
            condition: Input condition.
            time: Timestep.

        Returns:
            pred_noise: Predicted noise based on the given inputs.

        """
        # Encode conditional geometry
        encoded_condition = self.geometry_encoder(condition)[-1] # [Batch, Feature Channels, Height, Width]

        #[Batch, Feature Channels, Height * Width]
        encoded_condition = encoded_condition.reshape(*encoded_condition.shape[:2], -1) 

        # [Batch, sqe lenght, Feature Channels] 
        encoded_condition = encoded_condition.permute(0, 2, 1)

        # Predict noise based on the encoded condition
        pred_noise = self.condition_unet(
            sample = sample, 
            timestep = time,
            encoder_hidden_states = encoded_condition).sample
        
        return pred_noise
    


    