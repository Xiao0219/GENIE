# 文件: models/ReferenceNet_attention_xca.py
# 版本: 最终版 - 集成了 ResidualScalePredictor 创新点

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models.attention import BasicTransformerBlock, Attention

# 假设这个自定义的Block存在于同级目录的attention.py文件中
# from .attention import BasicTransformerBlock as _BasicTransformerBlock
# 为保证代码独立可运行，我们先定义一个占位符类
class _BasicTransformerBlock(torch.nn.Module):
    pass

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

# =====================================================================================
# 创新点核心：新的 ResidualScalePredictor 类
# =====================================================================================
class ResidualScalePredictor(nn.Module):
    """
    一个学习残差调节量的Scale预测器。
    它不直接预测scale，而是预测一个对基线值1.0的微调量delta。
    最终 scale = 1.0 + tanh(delta)，允许scale在[0，2]范围内浮动。
    """
    def __init__(self, input_channel):
        super().__init__()
        self.gelu = nn.GELU()
        
        # 网络结构可以保持与原来类似
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(input_channel, input_channel // 8, kernel_size=3, padding=1),
            nn.Conv2d(input_channel // 8, input_channel // 32, kernel_size=3, padding=1),
            nn.Conv2d(input_channel // 32, 1, kernel_size=3, padding=1)
        ])
        
        # 核心修改：初始化最后一层为0
        # 这保证了在训练刚开始时，网络输出的delta会非常接近0。
        nn.init.constant_(self.spatial_convs[-1].weight, 0)
        nn.init.constant_(self.spatial_convs[-1].bias, 0)

    def forward(self, x):
        """
        输入:
            x: (b, d, 2c)，d = h*w
        输出:
            (b, d, 1)，即每个像素的scale值，以1.0为中心浮动。
        """
        b, d, c = x.shape
        w = int(math.sqrt(d))
        h = d // w

        hidden_states = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        for i, spatial_layer in enumerate(self.spatial_convs):
            hidden_states = spatial_layer(hidden_states)
            if i < len(self.spatial_convs) - 1:
                hidden_states = self.gelu(hidden_states)
        
        # 此时的hidden_states是我们希望网络学习的微调量 delta
        delta = rearrange(hidden_states, "b c h w -> b (h w) c")

        # 使用tanh函数将delta的范围稳定在[-1, 1]之间，然后加到基线1.0上
        scale = 1.0 + torch.tanh(delta)
        
        return scale
# =====================================================================================

