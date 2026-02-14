# triton实现的rope

import torch
import triton
import triton.language as tl

class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        # 这里的cos和sin是预计算好的不同token位置的cos和sin，本程序要做的只是把计算好的cos和sin应用到x上
        # x: [SEQ_LEN, BATCH_SIZE, NUM_HEADS, HEAD_DIM]
        # cos/sin: [MAX_SEQ_LEN, HEAD_DIM // 2]
        SEQ_LEN = x.shape[0]
        BATCH_SIZE = x.shape[1]
        NUM_HEADS = x.shape[2]
        HEAD_DIM = x.shape[3]
        
        assert HEAD_DIM % 2 == 0, 'HEAD_DIM must be even'
        
        # 准备输出向量
        out = torch.empty_like(x)

        # 计算blocksize和grid，这里我们一次读入整个head dim，但是为了凑2的幂次，所以定义一个block size h用来方便kernel内的加载
        BLOCK_SIZE_H = triton.next_power_of_2(HEAD_DIM // 2)
        grid = lambda META: (
            triton.cdiv(SEQ_LEN, META["BLOCK_SIZE_SEQ"]),
            BATCH_SIZE,
            NUM_HEADS,
        )

        # 调用kernel
        _rope_kernel[grid](
            x, cos, sin, out, 
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            SEQ_LEN, HEAD_DIM, BLOCK_SIZE_H,
            BACKWARD = False
        )

        # 保存用于反向传播的参数和超参
        ctx.save_for_backward(cos, sin)
        ctx.BLOCK_SIZE_H = BLOCK_SIZE_H
        ctx.grid = grid

        return out
    
    @staticmethod
    def backward(ctx, d_out):
        # 反向传播和正向传播唯一的区别就是sin变个号。rope逆变换等价于旋转-theta
        cos, sin = ctx.saved_tensors
        BLOCK_SIZE_H = ctx.BLOCK_SIZE_H
        grid = ctx.grid

        SEQ_LEN = d_out.shape[0]
        BATCH_SIZE = d_out.shape[1]
        NUM_HEADS = d_out.shape[2]
        HEAD_DIM = d_out.shape[3]

        d_x = torch.empty_like(d_out)

        _rope_kernel[grid](
            d_out, cos, sin, d_x, 
            d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
            d_x.stride(0), d_x.stride(1), d_x.stride(2), d_x.stride(3),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            SEQ_LEN, HEAD_DIM, BLOCK_SIZE_H,
            BACKWARD = True
        )

        return d_x, None, None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_SEQ': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_SEQ': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_SEQ': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SEQ': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SEQ': 1024}, num_warps=8),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'], # 当 Seq Len 或 Head Dim 变化时重新 tune)
)
@triton.jit
def _rope_kernel(
    x, cos, sin, out, 
    stride_x_seq, stride_x_batch, stride_x_head, stride_x_head_dim,
    stride_out_seq, stride_out_batch, stride_out_head, stride_out_head_dim,
    stride_cos_seq, stride_cos_head_dim, stride_sin_seq, stride_sin_head_dim,
    SEQ_LEN, HEAD_DIM: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, 
    BACKWARD: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr
):
    seq_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    start_seq_idx = seq_block_idx * BLOCK_SIZE_SEQ

    # 计算offset
    offs_seq = tl.arange(0, BLOCK_SIZE_SEQ) + start_seq_idx
    # 先生成一个head dim一半长度的offset，然后通过*2和*2+1来得到x的偶数位和奇数位张量
    offs_dim_half = tl.arange(0, BLOCK_SIZE_H)

    # mask掉实际长度之外的block区域
    mask_seq = offs_seq < SEQ_LEN
    mask_dim = offs_dim_half < HEAD_DIM // 2
    # 两个维度广播
    mask = mask_seq[:, None] & mask_dim[None, :]

    # 得到张量偶数位和奇数位的ptr，shape：[BLOCK_SIZE_SEQ, BLOCK_SIZE_H // 2]
    offs_even = offs_dim_half * 2
    ptr_even = x + batch_idx * stride_x_batch + head_idx * stride_x_head + offs_seq[:, None].to(tl.int64) * stride_x_seq + offs_even[None, :].to(tl.int64) * stride_x_head_dim
    offs_odd = offs_dim_half * 2 + 1
    ptr_odd = x + batch_idx * stride_x_batch + head_idx * stride_x_head + offs_seq[:, None].to(tl.int64) * stride_x_seq + offs_odd[None, :].to(tl.int64) * stride_x_head_dim

    # 得到cos和sin的ptr，shape：[BLOCK_SIZE_SEQ, BLOCK_SIZE_H // 2]
    ptr_cos = cos + offs_seq[:, None].to(tl.int64) * stride_cos_seq + offs_dim_half[None, :].to(tl.int64) * stride_cos_head_dim
    ptr_sin = sin + offs_seq[:, None].to(tl.int64) * stride_sin_seq + offs_dim_half[None, :].to(tl.int64) * stride_sin_head_dim

    # load
    cos = tl.load(ptr_cos, mask=mask, other=0.0)
    sin = tl.load(ptr_sin, mask=mask, other=0.0)
    x_even = tl.load(ptr_even, mask=mask, other=0.0)
    x_odd = tl.load(ptr_odd, mask=mask, other=0.0)

    # 计算旋转（反向传播逆旋转）
    if BACKWARD:
        out_even = x_even * cos + x_odd * sin
        out_odd =  x_odd * cos - x_even * sin
    else:
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

    # 生成保存用ptr
    ptr_out_even = out + batch_idx * stride_out_batch + head_idx * stride_out_head + offs_seq[:, None].to(tl.int64) * stride_out_seq + offs_even[None, :].to(tl.int64) * stride_out_head_dim
    ptr_out_odd = out + batch_idx * stride_out_batch + head_idx * stride_out_head + offs_seq[:, None].to(tl.int64) * stride_out_seq + offs_odd[None, :].to(tl.int64) * stride_out_head_dim

    # save
    tl.store(ptr_out_even, out_even, mask=mask)
    tl.store(ptr_out_odd, out_odd, mask=mask)

# 提供api
def apply_rope(x, cos, sin):
    return TritonRoPE.apply(x, cos, sin)

def apply_rope_qk(q, k, cos, sin):
    return TritonRoPE.apply(q, cos, sin), TritonRoPE.apply(k, cos, sin)