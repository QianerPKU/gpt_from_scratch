# 组装attention和mlp层等（TP）

import torch
import torch.nn as nn

from core.parallel_state import get_tensor_model_parallel_world_size
from core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from core.rope_triton import apply_rope_qk
from core.myFlashAttn import apply_flash_attn
from core.position_embeddings import RotaryEmbedding
from core.mlp import ParallelMLP
from core.normalization import RMSNorm

class ParallelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.bias = config.bias

        # 获取tp信息
        world_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = self.num_heads // world_size # 每个tp的head数必须是整数，要求head数必须是tp数的整数倍

        # 初始化qkv投影层（注意输入的张量是scatter过的，输出的张量是切分过的）
        self.qkv_proj = ColumnParallelLinear(self.hidden_size, 3 * self.hidden_size, bias=self.bias)

        # 初始化输出投影（注意输入的张量是切分过的，注意输出的张量是scatter过的）
        self.out_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=self.bias)

        # 初始化rope的cos和sin缓存
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, hidden_states):
        # hidden_states: [S/P, B, H]
        qkv = self.qkv_proj(hidden_states)
        # qkv: [S, B, H/P * 3]

        seq_len, batch_size, _ = qkv.shape
        qkv = qkv.view(seq_len, batch_size, 3, self.num_heads_per_partition, self.head_dim)

        q, k, v = qkv.unbind(2)
        # q,k,v: [S, B, H/P]

        # 应用rope
        cos, sin = self.rotary_emb(q, seq_len = seq_len)
        q,k = apply_rope_qk(q, k, cos, sin)

        # 注意flash attention的qkv形状为：[B, H, S, D]，需要做转置
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # 调用flash attention
        output = apply_flash_attn(q, k, v)
        # output: [B, H, S, D]

        # 转置回来，并将H维度展平
        output = output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, -1)

        # 输出层
        output = self.out_proj(output)
        # output: [S, B, H]

        return output
    
class TransformerLayer(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # attention前的layer norm
        self.input_norm = RMSNorm(self.hidden_size, config.rms_norm_eps)

        # attention
        self.attention = ParallelAttention(config)

        # mlp前的layer norm
        self.post_attention_norm = RMSNorm(self.hidden_size, config.rms_norm_eps)

        # mlp
        self.mlp = ParallelMLP(config)

        # dropout（注意随机种子处理，这里先不加随机种子）
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states):
        # attention部分
        # 保存残差
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # MLP部分
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states