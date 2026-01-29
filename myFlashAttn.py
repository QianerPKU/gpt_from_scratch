import torch
from torch.profiler import profile, record_function, ProfilerActivity 
import triton
import triton.language as tl
import time

# torch实现的attention，用于结果对比
def ref_attn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, Q, K, V, casual_mask = True, dtype = torch.float16):
    # Q: [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    # K: [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    # V: [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    # O: [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    # torch.matmul默认将第一个张量的最后一个维度和第二个张量的倒数第二个维度进行矩阵乘法，前N-2个维度进行广播
    P = torch.matmul(Q,K.transpose(2,3)) / (HEAD_DIM ** 0.5)  # [B, NUM_HEADS, SEQ_LEN(Q), SEQ_LEN(K)]
    # torch.tril 返回矩阵的下三角部分（包括对角线），其余部分设为0，用于产生因果掩码
    if casual_mask:
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=Q.device))
        P = P.masked_fill(MASK == 0, float('-inf'))
    # 计算softmax后结果，softmax中的累加操作是fp32的，最后需要转回dtype
    P = torch.softmax(P, dim=-1).to(dtype) # [B, NUM_HEADS, SEQ_LEN(Q), SEQ_LEN(K)]
    O = torch.matmul(P, V)
    return O

# 测试torch实现与手写triton的结果一致
def test_flash_attn():
    # 测试参数
    BATCH_SIZE = 32
    NUM_HEADS = 12
    SEQ_LEN = 1024   
    HEAD_DIM = 64
    dtype = torch.float16
    device = torch.device('cuda')
    Q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype).requires_grad_()
    K = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype).requires_grad_()
    V = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype).requires_grad_()

    # ref前向传播
    O_ref = ref_attn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, Q, K, V, casual_mask=True, dtype=dtype)

    # triton前向传播
    O_tri = FlashAttn.apply(Q, K, V, True)

    # 比较结果
    assert torch.allclose(O_ref, O_tri, atol=1e-2)

    # ref反向传播
    dO = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype)
    O_ref.backward(dO)
    # 获取qkv的梯度
    dQ_ref, dK_ref, dV_ref = Q.grad, K.grad, V.grad

    # 清空梯度，因为梯度是累加的，不清空会影响后续结果
    Q.grad, K.grad, V.grad = None, None, None

    # triton反向传播
    O_tri.backward(dO)
    dQ_tri, dK_tri, dV_tri = Q.grad, K.grad, V.grad
    Q.grad, K.grad, V.grad = None, None, None

    # 比较结果
    assert torch.allclose(dQ_ref, dQ_tri, atol=5e-2)
    assert torch.allclose(dK_ref, dK_tri, atol=5e-2)
    assert torch.allclose(dV_ref, dV_tri, atol=5e-2)

# 性能测试
def profile_flash_attn():
    BATCH_SIZE = 4
    NUM_HEADS = 8                                                                                                                                                                                                                                                                
    SEQ_LEN = 1024                                                                                                                                                                                                                                                               
    HEAD_DIM = 64
    dtype = torch.float16
    device = torch.device('cuda')

    random_seed = 42
    torch.manual_seed(random_seed)

    def make_tensors():
        Q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
        K = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
        V = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype)
        return Q, K, V, dO

    # 包装 ref_attn 便于 compile
    def ref_attn_wrapper(Q, K, V):
        return ref_attn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, Q, K, V)

    # 编译版本 - 可选不同模式: "default", "reduce-overhead", "max-autotune"
    compiled_attn = torch.compile(ref_attn_wrapper, mode="default")

    # 预热（包括 compile 版本的编译）
    print("Warming up (torch.compile may take a while on first run)...")
    for _ in range(3):
        Q, K, V, dO = make_tensors()
        ref_attn_wrapper(Q, K, V).backward(dO)
        Q, K, V, dO = make_tensors()
        FlashAttn.apply(Q, K, V, True).backward(dO)
        Q, K, V, dO = make_tensors()
        compiled_attn(Q, K, V).backward(dO)  # 第一次会触发编译
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # 预热
    Q, K, V, dO = make_tensors()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O = FlashAttn.apply(Q, K, V, True)
        torch.cuda.synchronize()


    # ========== Triton Forward ==========
    Q, K, V, dO = make_tensors()
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O = FlashAttn.apply(Q, K, V, True)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Triton Flash Attn Forward")
    print("=" * 80)
    print(prof.key_averages().table())
    print(f"Time: {end - start}")

    # ========== Triton Backward ==========
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O.backward(dO)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Triton Flash Attn Backward")
    print("=" * 80)
    print(prof.key_averages().table())
    print(f"Time: {end - start}")

    # ========== Ref Forward ==========
    Q, K, V, dO = make_tensors()
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O = ref_attn_wrapper(Q, K, V)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Ref Attn Forward (eager)")
    print("=" * 80)
    print(prof.key_averages().table( ))
    print(f"Time: {end - start}")

    # ========== Ref Backward ==========
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O.backward(dO)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Ref Attn Backward (eager)")
    print("=" * 80)
    print(prof.key_averages().table( ))
    print(f"Time: {end - start}")

    # ========== Compiled Forward ==========
    Q, K, V, dO = make_tensors()
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O = compiled_attn(Q, K, V)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Ref Attn Forward (torch.compile)")
    print("=" * 80)
    print(prof.key_averages().table( ))
    print(f"Time: {end - start}")

    # ========== Compiled Backward ==========
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        O.backward(dO)
        torch.cuda.synchronize()
    end = time.perf_counter()
    print("=" * 80)
    print("Ref Attn Backward (torch.compile)")
    print("=" * 80)
    print(prof.key_averages().table( ))
    print(f"Time: {end - start}")

# tridon实现的flash attention
# 在torch中自定义的函数，总是需要继承torch.autograd.Function，并实现静态方法forward和backward
'''
从torch.autograd.Function继承的类需要重载两个静态方法：forward和backward。
forward方法输入模型的输入，输出模型的输出，并保存一些中间结果到ctx
backward方法输入模型的输出的梯度和ctx，输出模型输入的梯度，并计算所有需要的梯度进行保存
在调用forward的时候的语法是MyFunc.apply(inputs)，调用backward的时候是output.backward(grad_outputs)
'''
class FlashAttn(torch.autograd.Function):
    @staticmethod
    # 这里的ctx是用来在前向传播中保存下用来计算反向传播的中间结果。
    def forward(ctx, Q, K, V, casual_mask=True):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        #标准化缩放
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)
        # 检查QKV形状一致
        assert Q.shape == K.shape == V.shape
        # 定义输出张量，并自动继承Q的形状、device、dtype和layout（内存布局）
        O = torch.empty_like(Q)

        '''
        接下来需要用triton发送cuda kernel的调用指令

        triton调用cuda kernel的方式为，一次kernel调用的所有计算任务称为一个grid，一个grid中可以包含多个thread block，一个thread block中可以包含多个thread
        triton支持我们自定义grid和block，但是block内具体调用多少thread是由triton自动计算的，我们只需要管到block层级就行
        其中block是一个逻辑上的并行单元，一个block会发送给gpu硬件上的一个SM，但是一个SM可以同步处理多个block
        不同的block之间的内存无法互通，而一个block内部可以通过私有内存进行通信，这个通信的速度比访问global memory快非常多，因此我们要尽量减少读写global memory的次数
        同时，不同的block之间无法进行任何同步操作，但是block内部不同thread可以进行同步，因此不要尝试规定不同block完成任务的顺序

        我们将SEQ_LEN的Q拆分成若干个小块Q_block，每块大小为BLOCK_SIZE_Q，并发送给不同的thread block
        将SEQ_LEN的KV拆分成若干个小块KV_block，每块大小为BLOCK_SIZE_KV，并在每个Q的thread block中通过滑窗遍历计算attention score
        也就是说，我们会并行的执行不同batch、不同head、不同Q_block，一个并行block的内部会循环所有KV_block，并在一次循环内计算出对应Q_block的输出O
        '''
        
        '''
        发送指令到cuda kernel需要给定grid参数，描述的是需要发送多少个thread block，每个block有多少个thread由triton自动优化。
        block的坐标可以分成三个维度，分别是tl.program_id(0), tl.thread_id(1), tl.thread_id(2)。
        在我们当前的情况中，用第0个维度描述Q_block的index，用第1个维度描述batch和head展平后的index，第2个维度不需要。
        我们通过lambda函数来定义grid，这样triton可以通过autotune自动优化BLOCK_SIZE_Q。这里的cdiv表示ceil除法，BLOCK_SIZE_Q由triton自动优化
        '''

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), NUM_HEADS * BATCH_SIZE, 1
        )

        # 我们还需要保存一个中间结果：logsumexp，用来计算反向传播（在后面细讲）。每个Q对应一个标量M。这个量需要用float32来存，一切涉及累积的量都需要用float32来减少舍入误差
        # M [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        '''
        接下来，我们要发送cuda kernel指令到_attn_fwd kernel
        发送方式为function[grid](params)，这是triton规定的语法。实际上通过重载[]算符实现。
        通过params将所有张量和参数传入cuda kernel。注意这一步传过去的只是张量的指针，实际上张量的内容还停留在gpu的global memory（DRAM）中。
        除了传所有张量和其shape之外，还需要传入张量的stride参数，这个参数用来描述在展平的内存中，每个维度的步长（也就是每个维度相邻index之间在物理内存上相距多少元素，这个量可能会随着转置等操作改变）
        这个调用完成后，会把计算结果写入到O和M中
        '''
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            O=O,
            M=M,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            casual_mask=casual_mask,
            softmax_scale=softmax_scale,
        )

        # 接下来我们需要保存一下值用来反向传播
        ctx.save_for_backward(Q, K, V, M, O)
        # 存下grid来在反向传播阶段使用和前向传播相同的并行调度逻辑
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.casual_mask = casual_mask
        return O
    
    @staticmethod
    def backward(ctx, dO):
        '''
        计算反向传播的流程：
        
        对于前向传播：
        S = Q @ K^T * softmax_scale
        P = softmax(S)
        O = P @ V

        对于反向传播：
        我们保存了QKVO和M（logsumexp）用于反向传播
        我们需要先分块重计算P:
        S = Q @ K^T * softmax_scale
        P = exp(S - M)  # 这里的M是logsumexp
        梯度计算：
        dP = dO @ V^T
        dS = P * (dP - sum(dP * P)) # 这里的sum是对所有的K维度求和，乘法是逐元素乘
        引入一个统计量D
        D = sum(dP * P) = sum((dO @ V^T) * P) = sum(dO * (P @ V)) = sum(dO * O) # 关键推导
        通过提前计算D，可以让我们只用一层循环就可以计算出dQdK，因为D表示了softmax梯度反向传播中jacobian的所有非对角元项。
        化简得
        dS = P * (dO @ V^T - D)
        最后计算dQ, dK, dV:
        dQ = dS @ K * softmax_scale = P * (dO @ V^T - D[:,None]) @ K * softmax_scale
        dK = dS^T @ Q * softmax_scale = P^T * (dO @ V^T - D[None,:]) @ Q * softmax_scale = P^T * (V @ dO^T - D[None,:]) @ Q * softmax_scale
        dV = P^T @ dO
        具体算法见下kernel的实现
        '''
        # 从ctx中获取前向传播保存的张量
        Q, K, V, M, O = ctx.saved_tensors
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        softmax_scale = ctx.softmax_scale
        casual_mask = ctx.casual_mask

        # 定义D张量
        D = torch.empty_like(M)  # D [BATCH_SIZE, NUM_HEADS, SEQ_LEN]

        grid_precompute = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), NUM_HEADS * BATCH_SIZE, 1
        )

        # 先通过一个核函数来precompute D
        _attn_bwd_precompute[grid_precompute](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
        )

        #  通过一个kernel来计算dQ, dK, dV
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        '''
        这里我们并行的处理所有KV_block的梯度。对于一个固定的KV_block，我们需要遍历所有的Q_block来累加出dK和dV，具体公式为：
        dK = P^T * (V @ dO^T - D[None,:]) @ Q * softmax_scale
        dV = P^T @ dO
        S = Q @ K^T * softmax_scale
        P = exp(S - M)  # 这里的M是logsumexp
        '''

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_KV']), NUM_HEADS * BATCH_SIZE, 1
        )

        _attn_bwd_LoopQ[grid](
            Q=Q,
            K=K,
            V=V,
            dO=dO,
            dK=dK,
            dV=dV,
            D=D,
            M=M,
            casual_mask=casual_mask,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_dO_batch=dO.stride(0),
            stride_dO_head=dO.stride(1),
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            NUM_HEADS=NUM_HEADS,
            BATCH_SIZE=BATCH_SIZE,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            softmax_scale=softmax_scale,
        )

        '''
        这里我们并行的处理所有KV_block的梯度。对于一个固定的KV_block，我们需要遍历所有的Q_block来累加出dK和dV，具体公式为：
        dQ = dS @ K * softmax_scale = P * (dO @ V^T - D[:,None]) @ K * softmax_scale
        S = Q @ K^T * softmax_scale
        P = exp(S - M)  # 这里的M是logsumexp
        '''

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), NUM_HEADS * BATCH_SIZE, 1
        )

        _attn_bwd_LoopKV[grid](
            Q=Q,
            K=K,
            V=V,
            dO=dO,
            dQ=dQ,
            D=D,
            M=M,
            casual_mask=casual_mask,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_dO_batch=dO.stride(0),
            stride_dO_head=dO.stride(1),
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            NUM_HEADS=NUM_HEADS,
            BATCH_SIZE=BATCH_SIZE,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            softmax_scale=softmax_scale,
        )

        return dQ, dK, dV, None


# 接下来需要实现前向传播的kernel

# triton.autotune用来自动调优，triton会尝试不同的BLOCK_SIZE_Q和BLOCK_SIZE_KV，num_stageas表示流水线深度，目的是让load的同时compute
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 128}, num_warps=8, num_stages=3),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
# 下面是前向传播的kernel实现
# tl.constexpr告诉triton编译器这是一个常量，即对于不同的thread block来说，它的值是固定的
@triton.jit
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    casual_mask: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    # 在矩阵乘法中，经常保持收缩维度的尺寸大于等于参与计算的分块大小，这样可以更好的利用性能和减少SRAM浪费。通过搭配triton的autotune，自动剪枝出符合要求的尺寸，加快autotune的速度
    # tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # 计算当前block处理的是哪个batch和哪个head，以及处理的是哪个Q_block
    block_q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1) // NUM_HEADS
    head_idx = tl.program_id(1) % NUM_HEADS

    # 由于我们传递的张量QKV实际上是一个指针，并且是指向整个张量的头部的，所以需要根据上述三个idx和stride计算出目前block的起始位置offsets（注意要调用stride来计算而不是直接用BATCH_SIZE和NUM_HEADS，防止有意料之外的事情发生）
    # 注意，默认情况下整数是32位的，但gpu的内存地址是64位的，所以需要强制转换为64位防止整型溢出

    offset = batch_idx.to(tl.int64) * stride_Q_batch + head_idx.to(tl.int64) * stride_Q_head

    '''
    接下来我们调用triton.language中的make_block_ptr来构建滑窗，用来循环计算online softmax
    我们需要传入下列参数：
    1. base：整个滑窗运行的起始位置，传入一个指针
    2. shape：整个滑窗运行的区域的形状
    3. strides：滑窗每个维度的步长
    4. offsets：目前滑窗的窗口的起始位置
    5. block_shape：滑窗窗口的形状
    6. order：用来告诉triton每个维度的储存优先级，描述维度变化的快慢，从最快变化的维度到最慢变化的维度。这个参数的目的是告诉triton如何进行合并访问优化，因为gpu在如果可以一次访问相邻的一段内存，效率最高。
    '''

    Q_block_ptr = tl.make_block_ptr( # 挑选出的切片：Q[batch_idx][head_idx][block_index_q * BLOCK_SIZE_Q : block_index_q * BLOCK_SIZE_Q + BLOCK_SIZE_Q][:]
        base=Q + offset, # 确定batch和head的idx
        shape=(SEQ_LEN, HEAD_DIM), # 整个滑窗可以处理序列中所有的Q_block和KV_block的注意力
        strides=(stride_Q_seq, stride_Q_dim), # 每个维度的步长
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0), # 当前thread block执行的是哪个Q_block，其他的block_q_idx不在这个program中执行，因此这个滑窗实际上不会滑动
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # 每个滑窗的形状
        order=(1, 0), # head维度是变化最快的
    )

    # 为了方便内积，我们需要把K的形状进行转置
    K_block_ptr = tl.make_block_ptr( # 挑选出的切片：K[batch_idx][head_idx][:][:]
        base = K + offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0), # 我们需要从序列的一开始进行滑动，通过一个循环来遍历这个滑窗
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr( # 挑选出的切片：V[batch_idx][head_idx][:][:]
        base = V + offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr( # 挑选出的切片：O[batch_idx][head_idx][block_index_q * BLOCK_SIZE_Q : block_index_q * BLOCK_SIZE_Q + BLOCK_SIZE_Q][:]
        base = O + offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0), # 和Q的设定同理，这个滑窗实际上不会滑动
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    '''
    接下来我们要滑动KV滑块，循环计算并累加到O。
    具体算法是，对于每一个q，维护一个标量m表示循环到目前的KV_block时，遇到的最大的Q @ K^T的值，用来计算softmax中的偏置
    对于每一个q，维护一个标量l表示循环到目前的 exp(Q @ K^T - m) 的求和，注意这里的m是目前最大的Q @ K^T的值
    对于每一个q，维护一个累加器o表示循环到目前的exp(Q @ K^T - m)与V内积的结果，注意我们把归一化操作留到最后再做

    但实际上，我们按block来处理qkv，所以m是一个Q_block内所有q的最大值；l是一个Q_block内所有q的exp(Q @ K^T - m)的和，o是一个Q_block内所有q的注意力分数与V内积的结果
    m的shape是[BLOCK_SIZE_Q, 1]，l的shape是[BLOCK_SIZE_Q, 1]，o的shape是[BLOCK_SIZE_Q, HEAD_DIM]

    在每一步更新的时候，首先更新m，然后用更新后的m去修正l之前的项的和并且加上新的项，然后计算新的O：（遍历i）
    S_i = Q @ K_i^T * softmax_scale
    m_i = max(m_i-1, max(S_i)) 循环到目前的最大值
    P_i = exp(S_i - m_i)
    l_i = l_i-1 * exp(m_i-1 - m_i) + sum(P_i) 
    O_i = O_i-1 * exp(m_i-1 - m_i) + P_i @ V_i

    最后除以归一化系数：
    O_i = O_i / l_i
    '''
    
    # 这里需要再显式获得qkv在整个滑窗内的offset，即使已经有了block ptr，我们还需要这个offset来处理casual mask，待会就能看到用处
    # 通过tl.arange获得一串Q_block对应的idx列表
    offset_q = block_q_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # 通过tl.arange获得一串KV_block对应的idx列表
    offset_kv = tl.arange(0, BLOCK_SIZE_KV)

    # 在SHM中初始化m,l,O_block，注意使用float32来保证累加精度
    m = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    l = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # 这里加一个1.0是为了防止如果所有k都被mask掉了，不至于除以0。这里加1不会影响数值结果，因为初态m=-inf，l=1，下一次更新就会把这个1全部衰减掉（exp(-inf)）
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32) # 这里O_block创建为fp32类型，为了保证累加精度不丢失，但是最后存回O的时候会转回O的精度类型

    # 由于Q_block不需要滑动，是已经确定的，因此我们可以将其加载到shared memory中
    Q_block = tl.load(Q_block_ptr)

    '''
    接下来，处理casual mask的逻辑

    对于casual_mask=True的情况，需要区分三类情况，并用stage参数来区分：
    1. 当前的Q_block中所有token的index都大于等于当前KV_block中所有token的index时，我们称这种情况为stage=3
    2. 当前的Q_block中部分token的index大于等于部分KV_block中token的index时，我们称这种情况为stage=2
    3. 当前的Q_block中所有token的index都小于当前KV_block中所有token的index时，这样的情况我们直接不发送给cuda kernel从而节省计算
    对于casual_mask=False的情况，stage=1

    对于stage=1的情况，直接从头到尾遍历KV_block
    对于stage=3的情况，首先从头到最后一个满足Q_block中所有token位于KV_block的所有token之后的KV_block开始遍历，并且遍历时不需要考虑mask
    对于stage=2的情况，专门遍历正好位于对角线上的KV_block，这时再加上mask
    我们用一个函数来实现上述三种情况
    '''

    # 无论是casual_mask=True还是casual_mask=False，都要先执行无mask的遍历，只不过遍历的区间不一样
    stage = 3 if casual_mask else 1
    O_block, l, m = _attn_fwd_inner(
        O_block,
        l,
        m,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        block_q_idx,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        stage,
        offset_q,
        offset_kv,
        SEQ_LEN,
    )

    # 对于casual_mask=True的情况，再执行stage=2的遍历
    if stage == 3:
        O_block, l, m = _attn_fwd_inner(
            O_block,
            l,
            m,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_q_idx,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offset_q,
            offset_kv,
            SEQ_LEN,
        )

    # 乘以归一化系数并保存到O
    O_block = O_block / l[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty)) # 将O_block数据类型保存为O的数据类型

    # 计算logsumexp并保存到M，从而在反向传播时重计算softmax不需要再循环获得归一化系数
    m += tl.math.log(l)

    # 用指针m_ptr指向M中正确的batch、head、Q_block起始位置
    m_ptr = M + batch_idx * NUM_HEADS * SEQ_LEN + head_idx * SEQ_LEN + block_q_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    tl.store(m_ptr, m)

# 前向传播的分支函数，用来处理不同stage的遍历
@triton.jit
def _attn_fwd_inner(
    O_block,
    l,
    m,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_q_idx,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    stage: tl.constexpr,
    offset_q,
    offset_kv,
    SEQ_LEN: tl.constexpr,
):
    # 计算遍历的区间（token的idx区间）
    if stage == 3:
        lo, hi = 0, block_q_idx * BLOCK_SIZE_Q
    elif stage == 2:
        lo, hi = block_q_idx * BLOCK_SIZE_Q, block_q_idx * BLOCK_SIZE_Q + BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # 这一步显式告诉编译器，lo是BLOCK_SIZE_Q的整数倍，从而优化编译器生成加载向量的指令时的性能。编译器只需要知道起点是对齐的，因此hi不需要这样操作
    else:
        lo, hi = 0, SEQ_LEN

    # 先根据lo和hi把kv的block ptr移动到正确的起始位置上
    # 调用tl.advance用来滑动滑窗，每个维度的滑动距离用元组表示。注意这里K是转置过的
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    '''
    接下来遍历完成计算：
    S_i = Q @ K_i^T * softmax_scale
    m_i = max(m_i-1, max(S_i))
    P_i = exp(S_i - m_i)
    l_i = l_i-1 * exp(m_i-1 - m_i) + sum(P_i)
    O_i = O_i-1 * exp(m_i-1 - m_i) + P_i @ V_i
    '''

    for i in range(lo, hi, BLOCK_SIZE_KV):
        # 由于i要跟offset_kv相加，告诉编译器i是BLOCK_SIZE_KV的整数倍可以优化这里的加法
        i = tl.multiple_of(i, BLOCK_SIZE_KV)

        # 加载K_block，V_block（Q_block已经加载了，形状是 BLOCK_SIZE_Q * HEAD_DIM）
        K_block = tl.load(K_block_ptr) # HEAD_DIM * BLOCK_SIZE_KV
        V_block = tl.load(V_block_ptr) # BLOCK_SIZE_KV * HEAD_DIM

        # 计算S_i
        S_block = tl.dot(Q_block, K_block) * softmax_scale # BLOCK_SIZE_Q * BLOCK_SIZE_KV

        # 按不同stage加mask
        if stage == 2:
            mask = offset_q[:, None] >= i + offset_kv[None, :] # 下三角包括对角线为1，上三角为0。利用triton的广播机制生成
            S_block = tl.where(mask, S_block, -1.0e6) # 这里不使用float("-inf")是因为有可能导致计算出现NaN，-1e6已经足够小

        # 计算m_i
        m_new = tl.maximum(m, tl.max(S_block, axis=1)) # maximum是逐元素比较，max是归约
        # 计算P_i
        P_block = tl.math.exp(S_block - m_new[:, None])

        # 计算l_i
        l = l * tl.math.exp(m - m_new) + tl.sum(P_block, axis=1)

        # 计算O_i，为了加速矩阵乘法，先转到fp16（注意前面累加的时候不能转，必须保持为fp32）
        P_block = P_block.to(V_block.dtype)
        O_block = O_block * tl.math.exp(m - m_new)[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # 注意这里这种写法的意思就是，直接把P和V的乘法结果加在O上，但是更优化，因为不需要用中间变量来存储P和V的乘法结果

        # 更新m
        m = m_new

        # 更新指针
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

    return O_block, l, m

# 反向传播的precompute核函数实现
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_Q': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 256}, num_warps=8),
        # ... 更多组合
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def _attn_bwd_precompute(
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    dO, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    D, # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
):
    block_q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1) // NUM_HEADS
    head_idx = tl.program_id(1) % NUM_HEADS

    # 把O和dO的block ptr构建出来
    O_block_ptr = tl.make_block_ptr(
        base=O + batch_idx.to(tl.int64) * stride_O_batch + head_idx.to(tl.int64) * stride_O_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO + batch_idx.to(tl.int64) * stride_O_batch + head_idx.to(tl.int64) * stride_O_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # 加载dO_block和O_block
    dO_block = tl.load(dO_block_ptr)
    O_block = tl.load(O_block_ptr)

    # 计算D
    D_block = tl.sum(dO_block * O_block, axis=1) # BLOCK_SIZE_Q
    # 计算offset并将D_block保存到D的正确位置
    D_ptr = D + batch_idx.to(tl.int64) * NUM_HEADS * SEQ_LEN + head_idx.to(tl.int64) * SEQ_LEN + block_q_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    tl.store(D_ptr, D_block)

# 反向传播的核函数实现

# 遍历Q的核函数
@triton.autotune( # 这里BATCH_SIZE_Q和BATCH_SIZE_KV必须相等
    configs=[
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_Q': 32, 'BLOCK_SIZE_KV': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=3),
        # ... 更多组合
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def _attn_bwd_LoopQ( # 这里SEQ_LEN的(q)和(k)表示同标记的维度的idx是同步的。
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q), HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    dO, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q), HEAD_DIM
    dK, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    dV, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    D, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q)
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q)
    casual_mask: tl.constexpr,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_dO_batch,
    stride_dO_head,
    stride_dO_seq,
    stride_dO_dim,
    NUM_HEADS: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    batch_idx = tl.program_id(1) // NUM_HEADS
    head_idx = tl.program_id(1) % NUM_HEADS
    block_kv_idx = tl.program_id(0)

    # 建立block ptr
    '''
    公式：
    dQ = P * (dO @ V^T - D[:,None]) @ K * softmax_scale
    dK = P^T * (V @ dO^T - D[None,:]) @ Q * softmax_scale
    dV = P^T @ dO
    其中P = exp(Q @ K^T * softmax_scale - M[:,None])
    注意K, V的idx是同步的，Q, M, dO, D的idx是同步的
    '''
    Q_block_ptr = tl.make_block_ptr(
        base=Q + batch_idx.to(tl.int64) * stride_Q_batch + head_idx.to(tl.int64) * stride_Q_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + batch_idx.to(tl.int64) * NUM_HEADS * SEQ_LEN + head_idx.to(tl.int64) * SEQ_LEN,
        shape=(SEQ_LEN,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO + batch_idx.to(tl.int64) * stride_dO_batch + head_idx.to(tl.int64) * stride_dO_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_dO_seq, stride_dO_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    M_block_ptr = tl.make_block_ptr(
        base=M + batch_idx.to(tl.int64) * NUM_HEADS * SEQ_LEN + head_idx.to(tl.int64) * SEQ_LEN,
        shape=(SEQ_LEN,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + batch_idx.to(tl.int64) * stride_K_batch + head_idx.to(tl.int64) * stride_K_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K_seq, stride_K_dim),
        offsets=(block_kv_idx * BLOCK_SIZE_KV, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + batch_idx.to(tl.int64) * stride_V_batch + head_idx.to(tl.int64) * stride_V_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(block_kv_idx * BLOCK_SIZE_KV, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        base=dK + batch_idx.to(tl.int64) * stride_K_batch + head_idx.to(tl.int64) * stride_K_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K_seq, stride_K_dim),
        offsets=(block_kv_idx * BLOCK_SIZE_KV, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        base=dV + batch_idx.to(tl.int64) * stride_V_batch + head_idx.to(tl.int64) * stride_V_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(block_kv_idx * BLOCK_SIZE_KV, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # 初始化dK_block, dV_block，注意数据类型为fp32以保证累加精度，后面再转回fp16
    dK_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    # K_block, V_block是固定的，可以直接加载到SHM
    K_block = tl.load(K_block_ptr) # BLOCK_SIZE_KV * HEAD_DIM
    V_block = tl.load(V_block_ptr) # BLOCK_SIZE_KV * HEAD_DIM

    # 接下来需要处理casual mask的逻辑，和前向传播类似
    # 同理需要计算offset_q和offset_kv用来生成mask
    offset_q = tl.arange(0, BLOCK_SIZE_Q)
    offset_kv = tl.arange(0, BLOCK_SIZE_KV) + block_kv_idx * BLOCK_SIZE_KV
    stage = 3 if casual_mask else 1
    dK_block, dV_block = _attn_bwd_LoopQ_inner(
        dK_block,
        dV_block,
        K_block,
        V_block,
        Q_block_ptr,
        D_block_ptr,
        M_block_ptr,
        dO_block_ptr,
        block_kv_idx,
        softmax_scale,
        HEAD_DIM,
        batch_idx,
        head_idx,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        stage,
        SEQ_LEN,
        offset_q,
        offset_kv,
    )
    if stage == 3:
        dK_block, dV_block = _attn_bwd_LoopQ_inner(
            dK_block,
            dV_block,
            K_block,
            V_block,
            Q_block_ptr,
            D_block_ptr,
            M_block_ptr,
            dO_block_ptr,
            block_kv_idx,
            softmax_scale,
            HEAD_DIM,
            batch_idx,
            head_idx,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            SEQ_LEN,
            offset_q,
            offset_kv,
        )
    
    # 保存dK_block, dV_block到全局内存，转回原始dtype
    tl.store(dK_block_ptr, dK_block.to(dK.type.element_ty))
    tl.store(dV_block_ptr, dV_block.to(dV.type.element_ty))

@triton.jit
def _attn_bwd_LoopQ_inner(
    dK_block,
    dV_block,
    K_block,
    V_block,
    Q_block_ptr,
    D_block_ptr,
    M_block_ptr,
    dO_block_ptr,
    block_kv_idx,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    batch_idx,
    head_idx,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    stage: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    offset_q,
    offset_kv,
):
    # 计算遍历的区间（token的idx区间）
    if stage == 3:
        lo, hi = block_kv_idx * BLOCK_SIZE_KV + BLOCK_SIZE_KV, SEQ_LEN
    elif stage == 2:
        lo, hi = block_kv_idx * BLOCK_SIZE_KV, block_kv_idx * BLOCK_SIZE_KV + BLOCK_SIZE_KV
        lo = tl.multiple_of(lo, BLOCK_SIZE_KV)
    else:
        lo, hi = 0, SEQ_LEN

    # 先根据lo和hi把Q, dQ, D, M, dO 的block ptr移动到正确的起始位置上
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    D_block_ptr = tl.advance(D_block_ptr, (lo,))
    M_block_ptr = tl.advance(M_block_ptr, (lo,))
    dO_block_ptr = tl.advance(dO_block_ptr, (lo, 0))
    for i in range(lo, hi, BLOCK_SIZE_Q):
        '''
        计算公式：
        dK = P^T * (V @ dO^T - D[None,:]) @ Q * softmax_scale
        dV = P^T @ dO
        其中P = exp(Q @ K^T * softmax_scale - M[:,None])
        '''
        i = tl.multiple_of(i, BLOCK_SIZE_Q)
        # 加载Q_block，D_block, M_block, dO_block（未转置）
        Q_block = tl.load(Q_block_ptr) # BLOCK_SIZE_Q * HEAD_DIM
        D_block = tl.load(D_block_ptr) # BLOCK_SIZE_Q
        M_block = tl.load(M_block_ptr) # BLOCK_SIZE_Q
        dO_block = tl.load(dO_block_ptr) # BLOCK_SIZE_Q * HEAD_DIM

        # 计算S - M
        S_block = tl.dot(Q_block, tl.trans(K_block)) * softmax_scale - M_block[:, None] # BLOCK_SIZE_Q * BLOCK_SIZE_KV

        # 按不同stage加mask
        if stage == 2:
            mask = i + offset_q[:, None] >= offset_kv[None, :]
            S_block = tl.where(mask, S_block, -1.0e6)

        # 计算P
        P_block = tl.math.exp(S_block) # BLOCK_SIZE_Q * BLOCK_SIZE_KV

        # 累加dV_block
        # 为了加速矩阵乘法，先转到fp16（dO_block已经是fp16了）
        dV_block = tl.dot(tl.trans(P_block.to(dO_block.dtype)), dO_block, dV_block) # BLOCK_SIZE_KV * HEAD_DIM
        
        # 计算dP
        dP_block = tl.dot(dO_block, tl.trans(V_block))

        # 计算dS
        dS_block = (P_block * (dP_block - D_block[:, None]) * softmax_scale).to(Q_block.dtype)

        # 累加dK_block
        dK_block = tl.dot(tl.trans(dS_block), Q_block, dK_block) # BLOCK_SIZE_KV * HEAD_DIM

        # 更新指针
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_SIZE_Q, 0))
        D_block_ptr = tl.advance(D_block_ptr, (BLOCK_SIZE_Q,))
        M_block_ptr = tl.advance(M_block_ptr, (BLOCK_SIZE_Q,))
        dO_block_ptr = tl.advance(dO_block_ptr, (BLOCK_SIZE_Q, 0))

    return dK_block, dV_block

# 遍历KV的核函数
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_Q': 32, 'BLOCK_SIZE_KV': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_warps=4, num_stages=3),
        # ... 更多组合
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def _attn_bwd_LoopKV( # 这里SEQ_LEN的(q)和(k)表示同标记的维度的idx是同步的。
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q), HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    dO, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q), HEAD_DIM
    dQ, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(k), HEAD_DIM
    D, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q)
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN(q)
    casual_mask: tl.constexpr,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_dO_batch,
    stride_dO_head,
    stride_dO_seq,
    stride_dO_dim,
    NUM_HEADS: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    batch_idx = tl.program_id(1) // NUM_HEADS
    head_idx = tl.program_id(1) % NUM_HEADS
    block_q_idx = tl.program_id(0)

    # 建立block ptr
    '''
    公式：
    dQ = P * (dO @ V^T - D[:,None]) @ K * softmax_scale
    dK = P^T * (V @ dO^T - D[None,:]) @ Q * softmax_scale
    dV = P^T @ dO
    其中P = exp(Q @ K^T * softmax_scale - M[:,None])
    注意K, V的idx是同步的，Q, M, dO, D的idx是同步的
    '''
    Q_block_ptr = tl.make_block_ptr(
        base=Q + batch_idx.to(tl.int64) * stride_Q_batch + head_idx.to(tl.int64) * stride_Q_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + batch_idx.to(tl.int64) * NUM_HEADS * SEQ_LEN + head_idx.to(tl.int64) * SEQ_LEN,
        shape=(SEQ_LEN,),
        strides=(1,),
        offsets=(block_q_idx * BLOCK_SIZE_Q,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO + batch_idx.to(tl.int64) * stride_dO_batch + head_idx.to(tl.int64) * stride_dO_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_dO_seq, stride_dO_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    M_block_ptr = tl.make_block_ptr(
        base=M + batch_idx.to(tl.int64) * NUM_HEADS * SEQ_LEN + head_idx.to(tl.int64) * SEQ_LEN,
        shape=(SEQ_LEN,),
        strides=(1,),
        offsets=(block_q_idx * BLOCK_SIZE_Q,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + batch_idx.to(tl.int64) * stride_K_batch + head_idx.to(tl.int64) * stride_K_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K_seq, stride_K_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + batch_idx.to(tl.int64) * stride_V_batch + head_idx.to(tl.int64) * stride_V_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        base=dQ + batch_idx.to(tl.int64) * stride_Q_batch + head_idx.to(tl.int64) * stride_Q_head,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_q_idx * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # 初始化dQ_block，注意数据类型为fp32以保证累加精度，后面再转回fp16
    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Q_block, D_block, dO_block, M_block是固定的，可以直接加载到SHM
    Q_block = tl.load(Q_block_ptr)
    D_block = tl.load(D_block_ptr)
    dO_block = tl.load(dO_block_ptr)
    M_block = tl.load(M_block_ptr)

    # 接下来需要处理casual mask的逻辑，和前向传播类似
    # 同理需要计算offset_q和offset_kv用来生成mask
    offset_q = tl.arange(0, BLOCK_SIZE_Q) + block_q_idx * BLOCK_SIZE_Q
    offset_kv = tl.arange(0, BLOCK_SIZE_KV)
    stage = 3 if casual_mask else 1
    dQ_block = _attn_bwd_LoopKV_inner(
        dQ_block,
        K_block_ptr,
        V_block_ptr,
        Q_block,
        D_block,
        M_block,
        dO_block,
        block_q_idx,
        softmax_scale,
        HEAD_DIM,
        batch_idx,
        head_idx,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        stage,
        SEQ_LEN,
        offset_q,
        offset_kv,
    )
    if stage == 3:
        dQ_block = _attn_bwd_LoopKV_inner(
            dQ_block,
            K_block_ptr,
            V_block_ptr,
            Q_block,
            D_block,
            M_block,
            dO_block,
            block_q_idx,
            softmax_scale,
            HEAD_DIM,
            batch_idx,
            head_idx,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            SEQ_LEN,
            offset_q,
            offset_kv,
        )
    
    # 保存dQ_block到全局内存，转回原始dtype
    tl.store(dQ_block_ptr, dQ_block.to(dQ.type.element_ty))

@triton.jit
def _attn_bwd_LoopKV_inner(
    dQ_block,
    K_block_ptr,
    V_block_ptr,
    Q_block,
    D_block,
    M_block,
    dO_block,
    block_q_idx,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    batch_idx,
    head_idx,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    stage: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    offset_q,
    offset_kv,
):
    # 计算遍历的区间（token的idx区间）
    if stage == 3:
        lo, hi = 0, block_q_idx * BLOCK_SIZE_Q
    elif stage == 2:
        lo, hi = block_q_idx * BLOCK_SIZE_Q, block_q_idx * BLOCK_SIZE_Q + BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN

    # 先根据lo和hi把K, V的block ptr移动到正确的起始位置上
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for i in range(lo, hi, BLOCK_SIZE_KV):
        '''
        计算公式：
        dQ = P * (dO @ V^T - D[:,None]) @ K * softmax_scale
        其中P = exp(Q @ K^T * softmax_scale - M[:,None])
        '''
        i = tl.multiple_of(i, BLOCK_SIZE_KV)
        # 加载K_block，V_block（未转置）
        K_block = tl.load(K_block_ptr) # BLOCK_SIZE_KV * HEAD_DIM
        V_block = tl.load(V_block_ptr) # BLOCK_SIZE_KV * HEAD_DIM

        # 计算S - M
        S_block = tl.dot(Q_block, tl.trans(K_block)) * softmax_scale - M_block[:, None] # BLOCK_SIZE_Q * BLOCK_SIZE_KV

        # 按不同stage加mask
        if stage == 2:
            mask = offset_q[:, None] >= i + offset_kv[None, :]
            S_block = tl.where(mask, S_block, -1.0e6)

        # 计算P
        P_block = tl.math.exp(S_block) # BLOCK_SIZE_Q * BLOCK_SIZE_KV
        
        # 计算dP
        dP_block = tl.dot(dO_block, tl.trans(V_block))

        # 计算dS
        dS_block = (P_block * (dP_block - D_block[:, None]) * softmax_scale).to(Q_block.dtype)

        # 累加dQ_block
        dQ_block = tl.dot(dS_block, K_block, dQ_block) # BLOCK_SIZE_Q * HEAD_DIM

        # 更新指针
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_SIZE_KV, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

    return dQ_block

if __name__ == '__main__':
    test_flash_attn()
    profile_flash_attn()