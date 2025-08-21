import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import sys
sys.path.append("D:\\pycharm_project\\merged_tag_Psei\\experiments\\Enformer")
from model.attention import MultiHeadAttention  # 假设你已有该模块

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

class SoftmaxPooling1D(nn.Module):
    def __init__(self, channels, pool_size=2, w_init_scale=2.0):
        super().__init__()
        self.pool_size = pool_size
        self.logit_linear = nn.Linear(channels, channels, bias=False)
        self.logit_linear.weight.data.copy_(torch.eye(channels) * w_init_scale)

    def forward(self, x):  # x: (B, C, L)
        assert x.shape[-1] % self.pool_size == 0
        x = rearrange(x, "b c (l p) -> b l p c", p=self.pool_size)
        x = x * torch.softmax(self.logit_linear(x), dim=-2)
        x = torch.sum(x, dim=-2)
        return rearrange(x, "b l c -> b c l")

def conv_block(in_channels, out_channels, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        GELU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    )

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):  # x: (B, L, C)
        seq_len = x.shape[1]
        if seq_len < self.target_length:
            raise ValueError(f"Sequence length {seq_len} is less than target {self.target_length}")
        start = (seq_len - self.target_length) // 2
        end = start + self.target_length
        return x[:, start:end, :]

class Enformer_new(nn.Module):
    def __init__(self,
                 input_length=1024,
                 target_bins=8,
                 num_targets=5313,
                 channels=192,
                 num_heads=4,
                 num_transformer_layers=4):
        super().__init__()
        dropout_rate = 0.4
        assert channels % num_heads == 0

        self.stem = nn.Sequential(
            Rearrange("b l c -> b c l"),
            nn.Conv1d(4, channels, 15, padding=7),
            Residual(conv_block(channels, channels, 1)),
            SoftmaxPooling1D(channels, pool_size=2)  # 1024 → 512
        )

        # 继续做 pooling 直到达到 8 bins（1024 → 8 需要 7 次 pool_size=2）
        conv_blocks = []
        for _ in range(6):  # 再做6次，512→256→128→64→32→16→8
            conv_blocks.append(nn.Sequential(
                conv_block(channels, channels, 5),
                Residual(conv_block(channels, channels, 1)),
                SoftmaxPooling1D(channels, pool_size=2)
            ))
        self.conv_tower = nn.Sequential(*conv_blocks)

        attn_kwargs = {
            "input_dim": channels,
            "value_dim": channels // num_heads,
            "key_dim": 64,
            "num_heads": num_heads,
            "scaling": True,
            "attention_dropout_rate": 0.05,
            "relative_position_symmetric": False,
            "num_relative_position_features": channels // num_heads,
            "positional_dropout_rate": 0.01,
            "zero_initialize": True
        }

        def transformer_block():
            return nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(channels),
                    MultiHeadAttention(**attn_kwargs),
                    nn.Dropout(dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, channels * 2),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(channels * 2, channels),
                    nn.Dropout(dropout_rate)
                ))
            )

        self.transformer = nn.Sequential(
            Rearrange("b c l -> b l c"),
            *[transformer_block() for _ in range(num_transformer_layers)]
        )

        self.crop_final = TargetLengthCrop(target_bins)  # Crop 8 bins
        self.final_pointwise = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        self.head = nn.Sequential(
            nn.Linear(channels * 2, num_targets),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, 1024, 4)
        x = self.stem(x)                 # (B, C, 512)
        x = self.conv_tower(x)          # (B, C, 8)
        x = self.transformer(x)         # (B, 8, C)
        x = self.crop_final(x)          # (B, 8, C)
        x = self.final_pointwise(x)     # (B, 8, C*2)
        x = self.head(x)                # (B, 8, num_targets)
        return x
