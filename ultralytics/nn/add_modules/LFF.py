# Copyright (c) MCG-NKU. All rights reserved.
from typing import Sequence, Union

# import torch
# import torch.nn as nn
# from torch import Tensor
# 
# from mmyolo.registry import MODELS
# from mmdet.utils import OptConfigType
# from mmcv.cnn import ConvModule
#
# from ..utils import autopad
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

class LFFModule(nn.Module):
    """LFFModule
    Notice: The LFF code is currently not publicly available because the paper has not yet been officially published
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int, tuple[int]): The kernel size of this Module.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to None.
    """
