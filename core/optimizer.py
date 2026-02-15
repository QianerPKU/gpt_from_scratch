import torch
import torch.distributed as dist
import math
from core.parallel_state import get_data_parallel_group, get_data_parallel_group_rank, get_data_parallel_world_size

class DistributedOptimizer():
    '''
    实现了一个分布式的优化器，预分配grad_buffer，不同的gpu将自己负责的tp和dp部分得到的梯度数据重定向保存到grad_buffer中。
    同时预分配param_buffer，将不同gpu负责的参数的指针直接指向这部分预分配的空间。
    这个param_buffer和grad_buffer都是bf16的，是用来计算forward和backward用的，从而实现混合精度训练。
    分配fp32的master shard，是本rank负责更新的参数的fp32拷贝。真正的optimizer.step()发生在这个张量上。
    这个fp32的master shard是全精度的参数，也是我们最终想要保存和更新的参数。
    在更新时将grad_buffer做reduce-scatter，每个rank负责一部分的梯度更新。分配一个临时的bf16的切片张量reduced_grad_shard用来接收reduce scatter的结果。
    reduce-scatter得到的梯度切片reduced_grad_shard保存在master shard的grad中，然后对这个张量做step()
    每个rank只保存自己需要更新的参数切片对应的优化器参数。
    最后，将各个rank的fp32的参数切片all gather回param_buffer上。
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cls, # torch.optim.AdamW
        optimizer_kwargs: dict,
        clip_grad: float = 1.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model = model
        optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.dp_group = get_data_parallel_group()
        self.dp_rank = get_data_parallel_group_rank()
        self.dp_world_size = get_data_parallel_world_size()
        self.clip_grad = clip_grad
        self.dtype = dtype

        # 获取所有需要梯度的参数
        self.params = [p for p in model.parameters() if p.requires_grad]

        # 计算总参数量的大小（param和grad是一样的）
        total_numel = sum(p.numel() for p in self.params)

        # 确保总参数量能被dp world size整除，需要进行padding
        self.padding_size = (self.dp_world_size - total_numel % self.dp_world_size) % self.dp_world_size
        self.padded_numel = total_numel + self.padding_size

        # 每个rank负责的参数量
        self.shard_size = self.padded_numel // self.dp_world_size

        # 预分配param buffer和grad buffer
        self.param_buffer = torch.zeros(self.padded_numel, dtype=self.dtype, device=torch.cuda.current_device())

        self.grad_buffer = torch.zeros(self.padded_numel, dtype=self.dtype, device=torch.cuda.current_device())

        # 指针重定向，将param的data展平并copy到buffer中，然后再将param.data和param.grad指针指向buffer
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_buffer[offset:offset + numel].copy_(p.data.view(-1))
            p.data = self.param_buffer[offset:offset + numel].view(p.shape) # 注意需要view(p.shape)让原本展平的这一小段切片以原本形状返回指针
            p.grad = self.grad_buffer[offset:offset + numel].view(p.shape)
            offset += numel

        # 初始化fp32的参数切片（step真正发生的地方）
        # step作用的数据一定要是fp32的，要保证梯度更新的精度足够高
        shard_start = self.shard_size * self.dp_rank
        shard_end = shard_start + self.shard_size
        # 注意要转为fp32；注意要复制一份；注意要从计算图中detach；注意要requires_grad不然没有grad
        self.fp32_master_shard = self.param_buffer[shard_start:shard_end].to(dtype=torch.float32).clone().detach().requires_grad_(True)

        # 初始化fp32_master_shard上的优化器
        self.optimizer = optimizer_cls([self.fp32_master_shard], **self.optimizer_kwargs) # **dict
        
        # 初始化reduced_grad_shard用来接收reduce scatter的结果
        self.reduced_grad_shard = torch.zeros(self.shard_size, dtype=self.dtype, device=torch.cuda.current_device())

    def zero_grad(self):
        '''用来清零梯度，注意不能把grad设为none，不然我们重定向的指针就废了'''
        self.grad_buffer.zero_()

    def step(self):
        # loss backward已经填满了grad buffer，需要先reduce scatter到reduced_grad_shard
        dist.reduce_scatter_tensor(self.reduced_grad_shard, self.grad_buffer, group=self.dp_group)

        # 执行梯度裁剪
        if self.clip_grad is not None and self.clip_grad > 0.0:
            self._clip_grad_norm()

        # 将reduced_grad_shard的梯度更新到fp32_master_shard
        self.fp32_master_shard.grad = self.reduced_grad_shard.to(self.fp32_master_shard.dtype)

        # 更新fp32_master_shard，并清理梯度
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 将fp32_master_shard的梯度all gather到bf16的param buffer，注意先转回bf16
        dist.all_gather_into_tensor(self.param_buffer, self.fp32_master_shard.to(self.dtype), group=self.dp_group)
        # 由于模型的参数是指向param buffer的，所以all gather之后模型的参数也会更新，不用再做任何copy了

    def _clip_grad_norm(self):
        '''计算本地分片的梯度平方和，然后做all reduce得到全局梯度平方和，并计算缩放系数执行裁剪'''
        local_norm = torch.sum(self.reduced_grad_shard.to(torch.float32) ** 2) # 注意要转为fp32求平方和
        dist.all_reduce(local_norm, group=self.dp_group, op=dist.ReduceOp.SUM)
        global_norm = torch.sqrt(local_norm)
        if global_norm > self.clip_grad:
            self.reduced_grad_shard.mul_(self.clip_grad / (global_norm + 1e-6))

    def state_dict(self):
        '''保存优化器的状态字典'''
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        '''加载优化器的状态字典'''
        self.optimizer.load_state_dict(state_dict)