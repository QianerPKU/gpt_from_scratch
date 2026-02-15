'''
为每个rank分配到tp和dp的group。
这是一个dist的变量，在之后调用nccl的通讯指令时需要通过这个变量来确定自己的group和rank。
'''
import torch
import torch.distributed as dist

# 定义全局变量，用来描述当前TP DP的group信息
# group为一个torch.distributed.ProcessGroup 类的实例，通讯组
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP_RANK = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_RANK = None
_DATA_PARALLEL_WORLD_SIZE = None

# 初始化函数，用来初始化全局变量

def initialize_parallel_state(tensor_model_parallel_size,):
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP_RANK
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE

    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_RANK
    global _DATA_PARALLEL_WORLD_SIZE

    # 环境检查
    assert dist.is_initialized()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 必须整除
    assert world_size % tensor_model_parallel_size == 0

    # 计算dp维度
    data_parallel_size = world_size // tensor_model_parallel_size

    # 初始化tp groups
    # 循环所有group，但只有当前进程在循环到的ranks范围内时，才初始化全局变量
    for i in range(data_parallel_size):
        start_rank = i * tensor_model_parallel_size
        end_rank = (i + 1) * tensor_model_parallel_size
        ranks=list(range(start_rank, end_rank))
        # 这是一个同步操作，需要等到ranks中所有进程都初始化完成
        tensor_model_parallel_group = dist.new_group(ranks=ranks)

        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = tensor_model_parallel_group
            _TENSOR_MODEL_PARALLEL_GROUP_RANK = dist.get_rank(group=tensor_model_parallel_group)
            _TENSOR_MODEL_PARALLEL_WORLD_SIZE = dist.get_world_size(group=tensor_model_parallel_group)

    # 初始化dp groups
    for i in range(tensor_model_parallel_size):
        start_rank = i * data_parallel_size
        end_rank = (i + 1) * data_parallel_size
        ranks=list(range(start_rank, end_rank))
        data_parallel_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = data_parallel_group
            _DATA_PARALLEL_GROUP_RANK = dist.get_rank(group=data_parallel_group)
            _DATA_PARALLEL_WORLD_SIZE = dist.get_world_size(group=data_parallel_group)

# 清理环境
def destroy_model_parallel():
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None

# Getter接口
def get_tensor_model_parallel_group():
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_data_parallel_group():
    return _DATA_PARALLEL_GROUP

def get_tensor_model_parallel_group_rank():
    return _TENSOR_MODEL_PARALLEL_GROUP_RANK

def get_data_parallel_group_rank():
    return _DATA_PARALLEL_GROUP_RANK

def get_tensor_model_parallel_world_size():
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE

def get_data_parallel_world_size():
    return _DATA_PARALLEL_WORLD_SIZE