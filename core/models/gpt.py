# 组装好的gptmodel，输入是[B, S]，内容是每个token的embedding_id，输出是[B, S, V/P]，内容是logits（tp切分后的）
import torch
import torch.nn as nn

from core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from core.tensor_parallel.mappings import gather_from_sequence_parallel_region # 用来在输出头前做最后的gather
from core.transformer_layer import TransformerLayer
from core.normalization import RMSNorm

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 内部实现了scatter，所以输出形状是[S/P, B, H]
        self.embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # 多层transformer block
        self.layers = nn.ModuleList([TransformerLayer(i, config) for i in range(config.num_layers)])

        # 最终归一化
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 输出层（输出是切分后的，形状是[S, B, V/P]）计算logits
        self.output = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs):
        # inputs形状是[B, S]
        hidden_states = self.embedding(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        
        logits = self.output(hidden_states)
        return logits