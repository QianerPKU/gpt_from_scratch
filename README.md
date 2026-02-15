# GPT From Scratch

一个从零实现的 GPT 分布式训练框架，支持 **Tensor Parallelism (TP) + Data Parallelism (DP) + Sequence Parallelism (SP)**。

核心亮点：
- 手写 **Triton Flash Attention** 内核（含前向 + 反向传播），实现 Online Softmax 分块计算
- 手写 **Triton RoPE** 内核，高效应用旋转位置编码
- 自实现 **分布式优化器**，支持 bf16 计算 + fp32 主权重更新的混合精度训练
- 完整的 **Tensor Parallel** 实现：Column/Row 并行线性层、Vocab 并行 Embedding、Vocab 并行 Cross Entropy Loss
- **Sequence Parallelism**：在非 TP 计算区域（LayerNorm、Dropout）对 sequence 维度进行切分，减少显存占用

---

## 快速开始

### 环境依赖

- PyTorch（支持 NCCL 后端的分布式版本）
- Triton
- HuggingFace `datasets` 和 `transformers`（用于数据预处理）
- NumPy

### 第一步：预处理数据

使用 `tools/preprocess_data.py` 将文本数据集转换为二进制 token 文件：

```bash
python tools/preprocess_data.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-v1 \
    --tokenizer_name gpt2 \
    --output_dir ./data/wikitext_bin
```

该脚本会：
1. 通过 HuggingFace `datasets` 库下载 WikiText-103 数据集
2. 使用 GPT2 Tokenizer 进行分词，并在每段文本末尾添加 EOS token
3. 将 token id 序列保存为 `uint16` 类型的 `.bin` 二进制文件（节省 4 倍存储空间）
4. 分别生成 `train.bin` 和 `validation.bin`

### 第二步：启动训练

使用 `torchrun` 启动分布式训练：

```bash
# 单机 4 卡，TP=2（即 DP=2）
torchrun --nproc_per_node=4 train.py \
    --data-path ./data/wikitext_bin/train.bin \
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --batch-size 8 \
    --seq-len 1024 \
    --lr 3e-4 \
    --max-steps 100000 \
    --log-interval 10
```

```bash
# 单机单卡（TP=1，DP=1）
torchrun --nproc_per_node=1 train.py \
    --data-path ./data/wikitext_bin/train.bin \
    --tensor-model-parallel-size 1 \
    --batch-size 8 \
    --max-steps 100000
```

**主要参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-path` | （必填） | 预处理后的 `.bin` 数据文件路径 |
| `--tensor-model-parallel-size` | 1 | Tensor Parallel 并行度 |
| `--hidden-size` | 768 | 隐藏层维度 |
| `--num-layers` | 12 | Transformer 层数 |
| `--num-attention-heads` | 12 | 注意力头数 |
| `--batch-size` | 8 | 每个 DP rank 的 micro batch size |
| `--seq-len` | 1024 | 序列长度 |
| `--lr` | 3e-4 | 学习率 |
| `--max-steps` | 100000 | 最大训练步数 |
| `--clip-grad` | 1.0 | 梯度裁剪阈值 |

---

## 项目架构

```
gpt_from_scratch/
├── train.py                          # 主训练循环入口
├── tools/
│   └── preprocess_data.py            # 数据预处理脚本
├── core/
│   ├── models/
│   │   └── gpt.py                    # GPT 模型组装
│   ├── transformer_layer.py          # Transformer Block (Attention + MLP)
│   ├── mlp.py                        # SwiGLU MLP 层
│   ├── myFlashAttn.py                # Triton 手写 Flash Attention
│   ├── rope_triton.py                # Triton 手写 RoPE
│   ├── position_embeddings.py        # Rotary Embedding (cos/sin 预计算与缓存)
│   ├── normalization.py              # RMSNorm 归一化
│   ├── optimizer.py                  # 分布式优化器 (混合精度)
│   ├── data.py                       # 分布式数据加载
│   ├── parallel_state.py             # TP/DP 通信组初始化与管理
│   └── tensor_parallel/
│       ├── layers.py                 # TP 并行线性层 + Vocab 并行 Embedding
│       ├── loss.py                   # Vocab 并行 Cross Entropy Loss
│       └── mappings.py               # 通信原语 (gather/reduce-scatter/scatter)
└── data/
    └── wikitext_bin/                 # 预处理后的二进制数据
```

### 数据流与逻辑关系

```
preprocess_data.py                    train.py
       │                                │
  WikiText 文本                    初始化分布式环境
       │                           初始化 TP/DP 通信组 (parallel_state)
  GPT2 Tokenizer 分词              预热 Triton 内核
       │                                │
  保存为 .bin 文件 ──────────> 构建 GPTModel (models/gpt.py)
                                         │
                                ┌────────┴────────┐
                                │                  │
                         构建 DistributedOptimizer  构建 Dataloader
                           (optimizer.py)          (data.py)
                                │                  │
                                └────────┬─────────┘
                                         │
                                    训练循环
                                    ┌─────────┐
                                    │ forward  │ → logits
                                    │ loss     │ → VocabParallelCrossEntropy
                                    │ backward │ → 梯度填入 grad_buffer
                                    │ step     │ → reduce-scatter → clip → fp32 step → all-gather
                                    └─────────┘
```

**模型内部前向传播流程：**

```
input_ids [B, S]
    │
    ▼
VocabParallelEmbedding ──→ [S/P, B, H]    (词表并行查表 + reduce-scatter 到 sequence 切片)
    │
    ▼
TransformerLayer × N
    │  ├─ RMSNorm                          (在 sequence 切片上执行)
    │  ├─ ParallelAttention
    │  │    ├─ ColumnParallel QKV Proj     (gather sequence → 线性变换 → 切分 heads)
    │  │    ├─ RotaryEmbedding + RoPE      (Triton 内核应用旋转位置编码)
    │  │    ├─ Flash Attention             (Triton 内核, Online Softmax)
    │  │    └─ RowParallel Out Proj        (线性变换 → reduce-scatter 回 sequence 切片)
    │  ├─ Residual + Dropout
    │  ├─ RMSNorm
    │  ├─ ParallelMLP (SwiGLU)
    │  │    ├─ ColumnParallel gate_up      (gather → 线性变换)
    │  │    ├─ SiLU(gate) * up
    │  │    └─ RowParallel down            (线性变换 → reduce-scatter)
    │  └─ Residual + Dropout
    │
    ▼
RMSNorm → ColumnParallel Output ──→ logits [S, B, V/P]
```

---

## 各模块详细说明

### `train.py` — 主训练循环

训练入口，负责：
1. 初始化分布式环境（`torch.distributed`，NCCL 后端）
2. 调用 `parallel_state` 建立 TP/DP 通信组
3. 预热 Triton 内核（Flash Attention + RoPE），避免 autotune 首次运行时损坏输出
4. 构建 GPTModel、DistributedOptimizer、DataLoader
5. 执行标准训练循环：`zero_grad → forward → loss → backward → optimizer.step`

### `tools/preprocess_data.py` — 数据预处理

将 HuggingFace WikiText 数据集转换为训练所需的二进制格式：
- 使用 GPT2 Tokenizer 批量分词
- 每段文本末尾追加 EOS token
- 以 `uint16` 格式保存（GPT2 词表大小约 50257，`uint16` 范围 0-65535 足够覆盖）

### `core/data.py` — 分布式数据加载

- **`DistributedDataset`**：使用 `np.memmap` 内存映射读取 `.bin` 文件，避免将整个数据集加载到内存
- 将连续 token 序列按 `seq_len` 切段，每段构造 `(input, label)` 对（错位一个 token）
- **`create_distributed_dataloader`**：使用 `DistributedSampler`，按 DP 维度自动分片数据

### `core/models/gpt.py` — GPT 模型

组装完整的 GPT 模型：
- `VocabParallelEmbedding` → N × `TransformerLayer` → `RMSNorm` → `ColumnParallelLinear`（output head）
- 输入 `[B, S]` 的 token ids，输出 `[S, B, V/P]` 的 logits（TP 切分后的词表维度）

### `core/transformer_layer.py` — Transformer Block

**`ParallelAttention`**：
- QKV 投影使用 `ColumnParallelLinear`（输出被 TP 切分为每个 rank 负责的 heads）
- 应用 RoPE 旋转位置编码
- 调用 Triton Flash Attention
- 输出投影使用 `RowParallelLinear`（reduce-scatter 回 sequence 并行切片）

**`TransformerLayer`**：
- Pre-Norm 架构：`RMSNorm → Attention → Residual → RMSNorm → MLP → Residual`

### `core/mlp.py` — SwiGLU MLP

采用 **SwiGLU** 门控激活函数：
- `gate_up_proj`（ColumnParallel）：将 `gate` 和 `up` 两个投影合并为一个矩阵乘法，减少通信次数
- 激活：`SiLU(gate) * up`
- `down_proj`（RowParallel）：投影回原始维度并 reduce-scatter

### `core/normalization.py` — RMSNorm

- 在 fp32 精度下计算均方根归一化，避免低精度累加误差
- 计算完成后转回原始精度再乘以可学习权重

### `core/position_embeddings.py` — Rotary Embedding

- 预计算并缓存 `cos` 和 `sin` 值：`θ_i = base^(-2i/d)`，`cos(pos × θ_i)`，`sin(pos × θ_i)`
- 支持动态扩容：当序列长度超出缓存范围时自动重新计算

### `core/parallel_state.py` — 并行状态管理

管理 TP 和 DP 的进程组划分：
- 给定 `world_size` 和 `tp_size`，自动计算 `dp_size = world_size / tp_size`
- 创建 TP 组（同一 TP 组内的 rank 共同处理一个模型副本）和 DP 组（同一 DP 组内的 rank 持有相同模型分片）
- 提供全局 getter 接口供其他模块查询当前 rank 的组信息

---

## 核心实现思路

### 1. Tensor Parallelism + Sequence Parallelism

本项目采用 Megatron-LM 风格的 TP+SP 方案：

**ColumnParallelLinear**（用于 QKV 投影、MLP 上投影）：
- 权重按**输出维度**切分，每个 TP rank 持有 `[out/P, in]` 的权重切片
- 前向：先 **gather** sequence 并行切片为完整序列，再做矩阵乘法
- 反向：对梯度做 **reduce-scatter** 回 sequence 切片

**RowParallelLinear**（用于 Attention 输出投影、MLP 下投影）：
- 权重按**输入维度**切分，每个 TP rank 持有 `[out, in/P]` 的权重切片
- 前向：独立计算后 **reduce-scatter**（对 TP 维度求和 + 对 sequence 维度切分）
- 反向：**gather** 梯度

**VocabParallelEmbedding**：
- 词表按 TP rank 切分，每个 rank 只存储自己负责范围内的 embedding
- 对不在本 rank 范围内的 token 输出置零，最终 reduce-scatter 合并结果

通信原语（`core/tensor_parallel/mappings.py`）全部封装为 `torch.autograd.Function`，确保反向传播时自动执行互补的通信操作。

### 2. 分布式优化器（混合精度训练）

`core/optimizer.py` 实现了类似 Megatron-LM 的 `DistributedOptimizer`，核心设计：

**预分配 Buffer + 指针重定向**：
- 预分配连续的 `param_buffer`（bf16）和 `grad_buffer`（bf16）
- 将模型所有参数的 `.data` 和 `.grad` 指针重定向到 buffer 的对应区域
- 这样 backward 产生的梯度直接写入连续的 `grad_buffer`，无需额外拷贝

**混合精度**：
- 前向/反向计算在 **bf16** 下进行（param_buffer 存储 bf16 参数）
- 每个 DP rank 持有自己负责切片的 **fp32 master weights**（`fp32_master_shard`）
- 优化器的 `step()` 在 fp32 上执行，保证参数更新的数值精度

**更新流程**：
```
backward 完成 → grad_buffer (bf16, 完整)
    │
    ▼  reduce-scatter (跨 DP ranks)
reduced_grad_shard (bf16, 本 rank 的切片)
    │
    ▼  梯度裁剪 (all-reduce 求全局范数)
    │
    ▼  拷贝到 fp32_master_shard.grad
    │
    ▼  AdamW.step() (fp32 精度)
    │
    ▼  all-gather (跨 DP ranks)
param_buffer (bf16, 完整参数已更新)
```

### 3. Triton Flash Attention

`core/myFlashAttn.py` 从零实现了 Flash Attention 的前向和反向传播：

**前向传播 — Online Softmax**：
- 将 Q 按 block 分配给不同的 thread block 并行处理
- 每个 Q block 内部循环遍历 KV block，使用 Online Softmax 算法：
  - 维护 `m`（当前最大值）和 `l`（指数和）两个标量进行增量更新
  - 每步更新：`m_new = max(m, max(S))`，`l = l * exp(m-m_new) + sum(exp(S-m_new))`
  - 输出也同步缩放：`O = O * exp(m-m_new) + P @ V`
  - 最终归一化：`O = O / l`
- 保存 `logsumexp = m + log(l)` 用于反向传播重计算

**Causal Mask 优化（Stage 划分）**：
- **Stage 3**：Q block 完全在 KV block 之后 → 无需 mask，直接计算
- **Stage 2**：Q block 与 KV block 在对角线交叉 → 需要逐元素 mask
- Q block 完全在 KV block 之前的情况 → 直接跳过，不发送到 GPU

**反向传播**：
- 预计算辅助量 `D = rowsum(dO ⊙ O)`，避免在主循环中重复计算
- 分两个内核分别计算 `dQ`（遍历 KV blocks）和 `dK/dV`（遍历 Q blocks）
- 通过 `P = exp(S - logsumexp)` 重计算 softmax 结果，无需存储完整的 attention 矩阵

### 4. Triton RoPE

`core/rope_triton.py` 使用 Triton 实现旋转位置编码的应用：

- 将 head_dim 拆分为偶数位和奇数位，分别加载
- 前向：`x_even' = x_even * cos - x_odd * sin`，`x_odd' = x_even * sin + x_odd * cos`
- 反向：等价于逆旋转（旋转角取负），即 `sin` 变号即可，无需额外存储
- 使用 `triton.autotune` 自动调优 sequence 维度的 block size

### 5. Vocab Parallel Cross Entropy

`core/tensor_parallel/loss.py` 实现了分布式交叉熵损失计算：

- 数值稳定：先在各 TP rank 取 local max，再 all-reduce 得到 global max，减去后再计算 exp
- 局部 exp 求和 + target logit 提取后，**合并为一次 all-reduce** 通信（将两个张量 stack 后一起 reduce），减少通信次数
- 反向传播：`grad = softmax - one_hot(target)`，使用 `scatter_add_` 高效构造 one-hot 并原地减去
