# tp实现的mlp层，采用SwiGLU门控激活函数

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

class ParallelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.bias = config.bias
        self.intermediate_size = getattr(config, 'intermediate_size', self.hidden_size * 4)

        # 将gate和up线性层合成一个
        self.gate_up_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size * 2, bias=self.bias)

        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=self.bias)

    def forward(self, x):
        # 计算gate_up_proj并拆分
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU
        x = F.silu(gate) * up

        # 计算down_proj
        x = self.down_proj(x)

        return x