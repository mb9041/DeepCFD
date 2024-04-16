# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import get_down_block




class GeometryEncoder(ModelMixin, ConfigMixin):
    """
    The GeometryEncoder class is responsible for encoding the geometry information using a UNet-like architecture.

    Args:
        sample_size (Optional[Union[int, Tuple[int, int]]]): The size of the input sample. Defaults to None.
        in_channels (int): The number of input channels. Defaults to 3.
        down_block_types (Tuple[str, ...]): The types of down blocks to use in the architecture. Defaults to ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D").
        block_out_channels (Tuple[int, ...]): The number of output channels for each block. Defaults to (224, 448, 672, 896).
        layers_per_block (int): The number of layers per block. Defaults to 2.
        downsample_padding (int): The padding size for downsampling. Defaults to 1.
        downsample_type (str): The type of downsampling to use. Defaults to "conv".
        dropout (float): The dropout rate. Defaults to 0.0.
        act_fn (str): The activation function to use. Defaults to "silu".
        attention_head_dim (Optional[int]): The dimension of attention heads. Defaults to 8.
        norm_num_groups (int): The number of groups for normalization. Defaults to 32.
        norm_eps (float): The epsilon value for normalization. Defaults to 1e-5.
        resnet_time_scale_shift (str): The time scale shift for the ResNet blocks. Defaults to "default".
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
    ):
        super().__init__()

        self.sample_size = sample_size

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        The forward method of the GeometryEncoder.

        Args:
            sample (torch.FloatTensor): The input tensor with shape (batch, channel, height, width).

        Returns:
            Union[UNet2DOutput, Tuple]: If `return_dict` is True, an UNet2DOutput is returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        # 1. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 2. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=None, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=None)

            down_block_res_samples += res_samples

        return down_block_res_samples