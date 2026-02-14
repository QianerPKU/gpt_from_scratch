# 实现tp的cross entropy loss
import torch
import torch.distributed as dist
from core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_group_rank, get_tensor_model_parallel_world_size

class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        # logits: [sequence_length, batch_size, vocab_size / TP world size]
        # target: [sequence_length, batch_size]
        # 计算公式为：loss = log(sum(exp(logits - max(logits))) + max(logits) - target_logits
        # 首先获得本rank的局部max，然后做reduce-max得到全局max
        max_logits = torch.max(logits, dim=-1)[0]
        dist.all_reduce(max_logits, group=get_tensor_model_parallel_group(), op=dist.ReduceOp.MAX)
        logits = logits - max_logits.unsqueeze(-1)
        
        # 先计算本rank指数和，然后reduce-sum得到全局指数和
        exp_logits = torch.exp(logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1) # [sequence_length, batch_size]
        # 这里我们先不all reduce，而是和target logits一起all reduce，节省一次通讯
        # dist.all_reduce(sum_exp_logits, group=get_tensor_model_parallel_group(), op=dist.ReduceOp.SUM)

        # 提取target对应的logits。本rank只负责target位于范围内的token的logits计算，其他置零，并最终reduce-sum。逻辑同embedding部分（core.tensor_parallel.layers.VocabParallelEmbedding）
        # 获取当前进程负责的词表范围
        rank = get_tensor_model_parallel_group_rank()
        world_size = get_tensor_model_parallel_world_size()
        partition_size = logits.size(-1)
        vocab_start_index = rank * partition_size
        vocab_end_index = (rank + 1) * partition_size

        # 构造mask
        target_mask = (target >= vocab_start_index) & (target < vocab_end_index) # [sequence_length, batch_size]
        local_target = target - vocab_start_index # [sequence_length, batch_size]
        local_target[~target_mask] = 0

        # 提取对应维度logits
        # gather算符做的事情是：从local_target的dim=-1维度中拿到每一个target值，然后找到logits中对应index的值，然后放进dim=-1对应维度的位置。因此这里要先unsqueeze再squeeze
        target_logits = logits.gather(dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1) # [sequence_length, batch_size]
        # mask
        target_logits[~target_mask] = 0.0
        # 这里和sum_exp_logits一起all reduce
        # dist.all_reduce(target_logits, group=get_tensor_model_parallel_group(), op=dist.ReduceOp.SUM)
        loss_data = torch.stack([sum_exp_logits, target_logits], dim=-1)
        dist.all_reduce(loss_data, group=get_tensor_model_parallel_group(), op=dist.ReduceOp.SUM)
        sum_exp_logits, target_logits = loss_data.unbind(dim=-1)

        
        # 计算loss
        loss = torch.log(sum_exp_logits) - target_logits

        # 我们需要保存softmax的结果来做反向传播
        # 反向传播公式：dlogits = softmax(logits) - target_onehot
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        # 保存归一化后的softmax结果。保存target mask和local target避免重复计算
        ctx.save_for_backward(exp_logits, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, local_target = ctx.saved_tensors
        # softmax: [sequence_length, batch_size, vocab_size / TP world size]
        # target_mask: [sequence_length, batch_size]
        # local_target: [sequence_length, batch_size]
        # 反向传播的梯度只需要在target对应位置减1
        # 使用scatter_算符来构造one hot张量。target处为1，其余位置为0
        # 直接把target mask作为要scatter的值，这样mask为True的位置scatter的是1，False的位置scatter的是0等效于没有做任何操作
        values = target_mask.unsqueeze(-1).to(softmax.dtype)
        grad_input = softmax
        grad_input.scatter_add_(dim=-1, index=local_target.unsqueeze(-1), src=-values)

        return grad_input * grad_output.unsqueeze(-1), None