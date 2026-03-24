import torch
import torch.distributed as dist
import os
import argparse
import time
from dataclasses import dataclass

from core.data import create_distributed_dataloader
from core.optimizer import DistributedOptimizer
from core.models.gpt import GPTModel
from core.parallel_state import initialize_parallel_state
from core.profiling import PhaseLogAccumulator, StepPhaseProfiler
from core.tensor_parallel.loss import _VocabParallelCrossEntropy

PHASE_LOG_ORDER = (
    'data_wait',
    'h2d',
    'forward',
    'loss',
    'backward',
    'optimizer_rs',
    'clip_grad',
    'optimizer_step',
    'param_ag',
    'idle_sync',
    'step_core_wall',
)
MEMORY_LOG_ORDER = (
    'alloc_mb',
    'reserved_mb',
    'peak_alloc_mb',
    'peak_reserved_mb',
)


@dataclass
class GPTConfig:
    """模型配置，所有TransformerLayer/MLP/Attention需要的超参数都在这里"""
    vocab_size: int = 50304 # GPT2的50257 pad到64倍数，保证能被tp整除
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 1024
    intermediate_size: int = 3072 # MLP中间层大小，默认4*hidden_size
    rms_norm_eps: float = 1e-5
    bias: bool = False
    dropout_prob: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description='myMegatron GPT Training')

    # 模型参数
    parser.add_argument('--vocab-size', type=int, default=50304)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-attention-heads', type=int, default=12)
    parser.add_argument('--max-position-embeddings', type=int, default=1024)
    parser.add_argument('--intermediate-size', type=int, default=3072)

    # 训练参数
    parser.add_argument('--data-path', type=str, required=True,
                        help='预处理后的.bin数据文件路径')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每个DP rank的micro batch size')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--profile-phases', action='store_true',
                        help='启用 step 级 phase 计时和显存统计')

    # 并行参数
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)

    return parser.parse_args()


def warmup_triton(args):
    """预热 triton 内核，fp32 + bf16 的 forward + backward 都要预热"""
    from core.myFlashAttn import apply_flash_attn
    from core.rope_triton import apply_rope

    tp_size = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads
    head_dim = args.hidden_size // num_heads
    B, S = 2, min(args.seq_len, 256)

    # 只预热 bf16（训练实际使用的 dtype）
    # fp32 不需要预热：模型参数和计算全部在 bf16 下，且 fp32 的大 block 配置
    # 在 shared memory 较小的 GPU 上会超出硬件限制（如 99KB shared memory）
    dtype = torch.bfloat16
    for nh in set([num_heads // tp_size, num_heads]):
        q = torch.randn(B, nh, S, head_dim, device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn(B, nh, S, head_dim, device='cuda', dtype=dtype, requires_grad=True)
        v = torch.randn(B, nh, S, head_dim, device='cuda', dtype=dtype, requires_grad=True)
        o = apply_flash_attn(q, k, v)
        o.backward(torch.randn_like(o))

    for nh in set([num_heads // tp_size, num_heads]):
        x = torch.randn(S, B, nh, head_dim, device='cuda', dtype=dtype, requires_grad=True)
        cos = torch.randn(S, head_dim // 2, device='cuda', dtype=dtype)
        sin = torch.randn(S, head_dim // 2, device='cuda', dtype=dtype)
        out = apply_rope(x, cos, sin)
        out.backward(torch.randn_like(out))

    torch.cuda.synchronize()


def reduce_phase_metrics(phase_totals, memory_max):
    phase_tensor = torch.tensor(
        [phase_totals.get(name, 0.0) for name in PHASE_LOG_ORDER],
        device='cuda',
        dtype=torch.float64,
    )
    memory_tensor = torch.tensor(
        [memory_max.get(name, 0.0) for name in MEMORY_LOG_ORDER],
        device='cuda',
        dtype=torch.float64,
    )
    dist.all_reduce(phase_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
    phase_tensor /= dist.get_world_size()
    return (
        {name: phase_tensor[idx].item() for idx, name in enumerate(PHASE_LOG_ORDER)},
        {name: memory_tensor[idx].item() for idx, name in enumerate(MEMORY_LOG_ORDER)},
    )


def format_phase_log(phase_times, memory_stats):
    phase_summary = " | ".join(
        f"{name}:{phase_times[name]:.1f}ms"
        for name in PHASE_LOG_ORDER
    )
    memory_summary = (
        f"alloc:{memory_stats['alloc_mb']:.0f}MB | "
        f"reserved:{memory_stats['reserved_mb']:.0f}MB | "
        f"peak_alloc:{memory_stats['peak_alloc_mb']:.0f}MB | "
        f"peak_reserved:{memory_stats['peak_reserved_mb']:.0f}MB"
    )
    return phase_summary, memory_summary


def main():
    '''主训练循环'''

    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    args = parse_args()

    # 初始化TP/DP通信组
    initialize_parallel_state(args.tensor_model_parallel_size)

    if rank == 0:
        print(f"World size: {world_size}, TP size: {args.tensor_model_parallel_size}, "
              f"DP size: {world_size // args.tensor_model_parallel_size}")

    # 预热 triton 内核（flash attention + RoPE），fp32 和 bf16 都要预热
    # triton autotune 首次运行时会 benchmark 多个配置，可能损坏输出缓冲区
    warmup_triton(args)

    # 构建模型配置和模型
    config = GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        intermediate_size=args.intermediate_size,
    )
    # 模型参数在各层__init__中已经分配到CUDA上
    model = GPTModel(config)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params (per TP rank): {total_params / 1e6:.1f}M")

    # 构建分布式优化器
    optimizer = DistributedOptimizer(
        model=model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'betas': (0.9, 0.95),
        },
        clip_grad=args.clip_grad,
    )

    # 构建数据加载器
    dataloader, sampler = create_distributed_dataloader(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if rank == 0:
        print(f"Dataset loaded: {len(dataloader.dataset)} samples, "
              f"{len(dataloader)} batches/epoch")
        print(f"Starting training for {args.max_steps} steps...")

    # 训练循环
    model.train()
    step = 0
    epoch = 0
    log_loss = 0.0
    log_start_time = time.time()
    phase_profiler = StepPhaseProfiler(enabled=args.profile_phases)
    phase_accumulator = PhaseLogAccumulator() if args.profile_phases else None

    while step < args.max_steps:
        sampler.set_epoch(epoch)
        data_iter = iter(dataloader)

        while True:
            data_wait_start = time.perf_counter() if args.profile_phases else None
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            if step >= args.max_steps:
                break

            # 清零梯度（不能设None，因为grad指针指向grad_buffer）
            optimizer.zero_grad()

            if args.profile_phases:
                phase_profiler.start_step()
                data_wait_ms = (time.perf_counter() - data_wait_start) * 1000.0
                phase_profiler.add_cpu_phase('data_wait', data_wait_ms)

            phase_name = phase_profiler.start_gpu_phase('h2d') if args.profile_phases else None
            inputs = batch['input'].cuda() # [B, S],token ids
            labels = batch['label'].cuda() # [B, S],shifted targets
            if args.profile_phases:
                phase_profiler.end_gpu_phase(phase_name)

            # 前向传播
            phase_name = phase_profiler.start_gpu_phase('forward') if args.profile_phases else None
            logits = model(inputs) # [S, B, V/P]
            if args.profile_phases:
                phase_profiler.end_gpu_phase(phase_name)

            # 计算TP并行的交叉熵损失
            # labels需要从[B, S]转为[S, B]以匹配logits的形状
            phase_name = phase_profiler.start_gpu_phase('loss') if args.profile_phases else None
            labels_t = labels.transpose(0, 1).contiguous()  # [S, B]
            loss = _VocabParallelCrossEntropy.apply(logits, labels_t)  # [S, B]
            loss = loss.mean()
            if args.profile_phases:
                phase_profiler.end_gpu_phase(phase_name)

            # 反向传播：梯度自动累积到grad_buffer中
            phase_name = phase_profiler.start_gpu_phase('backward') if args.profile_phases else None
            loss.backward()
            if args.profile_phases:
                phase_profiler.end_gpu_phase(phase_name)

            # 优化器步骤：reduce-scatter梯度 -> clip -> fp32 step -> all-gather参数
            optimizer.step(phase_profiler if args.profile_phases else None)

            if args.profile_phases:
                phase_times, memory_stats = phase_profiler.finalize_step()
                phase_accumulator.update(phase_times, memory_stats)

            step += 1
            log_loss += loss.item()

            # 日志输出（仅rank 0打印）
            if step % args.log_interval == 0:
                avg_loss = log_loss / args.log_interval
                elapsed = time.time() - log_start_time
                tokens_per_sec = (args.log_interval * args.batch_size * args.seq_len
                                  * (world_size // args.tensor_model_parallel_size)  # DP size
                                  / elapsed)

                if rank == 0:
                    print(f"Step {step:>6d}/{args.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Tokens/s: {tokens_per_sec:.0f} | "
                          f"Time: {elapsed:.1f}s")
                if args.profile_phases:
                    avg_phase_times, max_memory_stats = reduce_phase_metrics(
                        phase_accumulator.average_times(),
                        phase_accumulator.memory_max,
                    )
                    if rank == 0:
                        phase_summary, memory_summary = format_phase_log(
                            avg_phase_times,
                            max_memory_stats,
                        )
                        print(f"Phase Avg | {phase_summary}")
                        print(f"Memory Max | {memory_summary}")
                    phase_accumulator.reset()

                log_loss = 0.0
                log_start_time = time.time()

        epoch += 1

    if args.profile_phases and phase_accumulator.steps > 0:
        avg_phase_times, max_memory_stats = reduce_phase_metrics(
            phase_accumulator.average_times(),
            phase_accumulator.memory_max,
        )
        if rank == 0:
            phase_summary, memory_summary = format_phase_log(
                avg_phase_times,
                max_memory_stats,
            )
            print(f"Phase Avg | {phase_summary}")
            print(f"Memory Max | {memory_summary}")

    # 训练结束
    if rank == 0:
        print("Training finished.")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
