'''提供了可以forward和backward的通讯操作封装'''
import torch
import torch.distributed as dist
from torch.autograd import Function

from core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_group_rank

# 基础通讯函数（先写基础通讯函数，再包装成torch.autograd.Function）

# 将第一个维度做gather
def _gather_along_first_dim(input_):
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    # 计算输出的shape
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
    return output
    
# 将第一个维度做reduce scatter
def _reduce_scatter_along_first_dim(input_):
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    dist.reduce_scatter_tensor(output, input_.contiguous(), group=group)
    return output

# 将第一个维度切分（不做reduce，直接切分，应用于embedding后）
def _split_along_first_dim(input_):
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    input_list = torch.chunk(input_, world_size, dim=0)
    rank = get_tensor_model_parallel_group_rank()
    return input_list[rank].contiguous()

# autograd function封装

class _GatherFromSequenceParallelRegion(Function):
    '''
    用于SP时线性变换的输入时，将第一个维度做gather，然后再切分进行线性变换（此时的输入的sequence维度已被切分）。
    当反向传播时，要将切分的结果reduce，并scatter到sequence维度。
    '''
    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim(grad_output)
    
class _ReduceScatterToSequenceParallelRegion(Function):
    '''
    用于SP时线性变换的输出，将第一个维度做reduce scatter。
    反向传播时，直接将结果gather。
    '''
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)
    
class _ScatterToSequenceParallelRegion(Function):
    '''
    当输入的是完整序列，需要在sequence维度上切分，然后做layernorm或者dropout等。是整个SP流程的入口。
    反向传播时是gather。
    '''
    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)
    
# public api，用来调用前向传播
def gather_from_sequence_parallel_region(input_):
    '''
    用于SP时线性变换的输入时，将第一个维度做gather，然后再切分进行线性变换（此时的输入的sequence维度已被切分）。
    当反向传播时，要将切分的结果reduce，并scatter到sequence维度。
    '''
    return _GatherFromSequenceParallelRegion.apply(input_)

def reduce_scatter_to_sequence_parallel_region(input_):
    '''
    用于SP时线性变换的输出，将第一个维度做reduce scatter。
    反向传播时，直接将结果gather。
    '''
    return _ReduceScatterToSequenceParallelRegion.apply(input_)

def scatter_to_sequence_parallel_region(input_):
    '''
    当输入的是完整序列，需要在sequence维度上切分，然后做layernorm或者dropout等。是整个SP流程的入口。
    反向传播时是gather。
    '''
    return _ScatterToSequenceParallelRegion.apply(input_)