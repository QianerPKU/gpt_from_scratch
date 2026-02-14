# 标准化层，例如RMSnorm

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # 需要用fp32计算二阶矩，因此先保存输入的dtype最后再转回来
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算RMS
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt是先开根号再取倒数
        x = x * torch.rsqrt(rms + self.eps)
        # 转回原精度再乘以权重
        x = x.to(input_dtype)
        x = x * self.weight
        return x