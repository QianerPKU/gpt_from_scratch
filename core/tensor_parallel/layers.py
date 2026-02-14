# 这个程序用来实现tp相关的线性层，例如列并行线性层和行并行线性层和词表并行

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from core.tensor_parallel import mappings
from core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group_rank
)

# 列并行，用在attention的QKV投影上，以及MLP的第一个线性层
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, init_method=init.xavier_uniform_):
        super(ColumnParallelLinear, self).__init__()

        # 这里的input_size和output_size都是没切片的尺寸
        self.input_size = input_size
        self.output_size = output_size

        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_group_rank()

        # 计算每个rank的输出形状（切片）
        self.output_size_per_partition = self.output_size // world_size

        # 初始化权重
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size, device=torch.cuda.current_device(), dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, device=torch.cuda.current_device(), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        # 这里需要随机初始化，如果需要随机种子的话，要确保每张卡的随机种子不同
        init_method(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input_):
        # 输入的向量在sequence维度上是切分的，需要先gather拼成完整的
        input_parallel  = mappings.gather_from_sequence_parallel_region(input_)

        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        return output_parallel
    
# 列并行，用在attention的输出投影，以及MLP的最后一个线性层
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, init_method=init.xavier_uniform_):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size  

        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_group_rank()

        # 计算每个切片的输入形状
        self.input_size_per_partition = self.input_size // world_size

        # 初始化权重
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition, device=torch.cuda.current_device(), dtype=torch.float32))

        # 注意，本层中的bias是指reduce后加上的bias
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, device=torch.cuda.current_device(), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        # 这里需要随机初始化，如果需要随机种子的话，要确保每张卡的随机种子不同
        init_method(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
            
    def forward(self, input_):
        # 列并行，每个rank的计算是独立的，最终reduce scatter
        # 这里线性层先不加bias，bias在最后加
        output_parallel = F.linear(input_, self.weight)
        
        output = mappings.reduce_scatter_to_sequence_parallel_region(output_parallel)

        # 加上bias
        if self.bias is not None:
            output = output + self.bias

        return output

# 词表并行，即看对应token的idx是不是在本rank负责范围内，如果是就查表，否则置零，最后reduce scatter
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_uniform_):
        super(VocabParallelEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_group_rank()

        # 确保词表大小是能被world_size整除
        assert self.num_embeddings % world_size == 0, "Vocab size must be divisible by tensor model parallel size"

        # 计算该rank的词表范围
        self.vocab_start_index = self.num_embeddings // world_size * rank
        self.vocab_end_index = self.num_embeddings // world_size * (rank + 1)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # 初始化权重
        self.weight = nn.Parameter(torch.zeros(self.num_embeddings_per_partition, self.embedding_dim, device=torch.cuda.current_device(), dtype=torch.float32))

        # 这里需要随机初始化，如果需要随机种子的话，要确保每张卡的随机种子不同
        init_method(self.weight)

    def forward(self, input_ids):
        # input_ids的形状是Batch_size, Sequence_length
        # 建立mask，用来判断每个token的id是否在本rank的范围内，不在范围内的先当作id是0（防止越界），最后再乘以0
        input_mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        # 计算本rank内的编号
        input_ids = input_ids - self.vocab_start_index
        # maskfill（注意要对mask取反，因为我们取的mask是符合条件为True的）
        input_ids = input_ids.masked_fill(~input_mask, 0)

        output_parallel = F.embedding(input_ids, self.weight)

        # maskfill
        output_parallel = output_parallel.masked_fill(~input_mask.unsqueeze(-1), 0.0)

        # reduce之前要把sequence转置到第一个维度（别忘了连续化）
        # 这里要先让形状是B，S再变成S，B是因为F.embedding返回的是[B, S, E]，如果要进一步优化可以做算子融合
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = mappings.reduce_scatter_to_sequence_parallel_region(output_parallel)

        return output