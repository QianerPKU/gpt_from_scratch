'''分布式数据加载器，使用DistributedSampler实现，使用memmap加载'''
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from core.parallel_state import get_data_parallel_world_size, get_data_parallel_group_rank

class DistributedDataset(Dataset):
    def __init__(self, data_path, seq_len):
        '''
        Args:
            data_path (str): 数据文件路径
            seq_len (int): 上下文窗口大小
        '''
        self.seq_len = seq_len

        try:
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        except FileNotFoundError:
            print(f"File {data_path} not found.")
            raise

        # 计算总样本数（丢弃末端）
        self.num_samples = (len(self.data) - 1) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''用来访问第index段数据'''
        start_idx = index * self.seq_len
        end_idx = start_idx + self.seq_len + 1 # 构造样本对时会短一个，所以多加载一个

        chunk = self.data[start_idx:end_idx]
        
        # 转换为pytorch tensor(int64)
        chunk = torch.from_numpy(chunk.astype(np.int64))

        # 构造输入和标签（错位）
        return {
            'input': chunk[:-1],
            'label': chunk[1:]
        }
    
def create_distributed_dataloader(
    data_path,
    seq_len,
    batch_size,
    num_workers,
):
    '''创建支持ddp的dataloader'''
    dataset = DistributedDataset(data_path, seq_len)

    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=get_data_parallel_world_size(),
        rank=get_data_parallel_group_rank(),
        shuffle=True,
        drop_last=True, # 若最后一个batch填不满，丢弃
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler