'''用来计算cos和sin，并传给rope直接使用。cos和sin只需要计算一次就可以缓存到buffer里面，并且在seq_len超出已缓存的范围时重新动态计算'''

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    '''用来计算cos和sin，并传给rope直接使用。cos和sin只需要计算一次就可以缓存到buffer里面，并且在seq_len超出已缓存的范围时重新动态计算'''
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        if device is None:
            device = torch.cuda.current_device()

        # rope计算公式：freq_i = base ^ (-2i / head_dim) 这里的2i指的是head_dim第2i维和2i+1维转的角度
        # 注意精度要为float32，保证三角函数的计算精度
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        freqs = freqs.to(torch.float32)

        # 保存到buffer
        self.register_buffer('freqs', freqs)

        # 初始化缓存，一开始就缓存好max_position_embeddings个cos和sin
        self.max_seq_len_cached = 0
        self._set_cos_sin_cache(self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        # 缓存seq_len个cos和sin
        self.max_seq_len_cached = seq_len

        # 生成位置索引
        t = torch.arange(seq_len, dtype=torch.float32, device=self.freqs.device)

        # 计算cos和sin（外积）
        thetas = torch.outer(t, self.freqs)
        # thetas: [seq_len, head_dim // 2]
        cos = thetas.cos().to(dtype=torch.float32)
        sin = thetas.sin().to(dtype=torch.float32)

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            # 超过已缓存的最大长度，扩容
            self._set_cos_sin_cache(seq_len, x.device, torch.float32)

        # 直接返回切片
        return self.cos_cached[:seq_len, :].to(x.dtype), self.sin_cached[:seq_len, :].to(x.dtype)