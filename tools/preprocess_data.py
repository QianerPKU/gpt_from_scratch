'''负责把文本数据转换成numpy的二进制格式'''
import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def preprocess_wikitext(args):
    print(f"Loading {args.dataset_name} dataset...")
    # 使用 HuggingFace datasets 库直接加载
    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    
    # 确保有 eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    def process_split(split_name):
        print(f"Processing {split_name} split...")
        data = dataset[split_name]
        
        # 1. 批量分词 (并行加速)
        # 我们只关心 'text' 列，并且过滤掉空行
        text_column = [t for t in data['text'] if len(t) > 0]
        
        token_ids = []
        print(f"Tokenizing {len(text_column)} lines...")
        
        # 使用 batch_encode_plus 加速
        batch_size = 1000
        for i in tqdm(range(0, len(text_column), batch_size)):
            batch = text_column[i : i + batch_size]
            # WikiText 通常在每行末尾加 EOS 表示段落/句子结束
            # 注意：这里简单的处理方式，生产环境可能需要更复杂的文档拼接逻辑
            encodings = tokenizer(batch, add_special_tokens=False)
            
            for seq in encodings['input_ids']:
                token_ids.extend(seq)
                token_ids.append(tokenizer.eos_token_id) # 添加 EOS

        # 2. 转换为 Numpy 数组
        # GPT2 vocab size ~50257，可以用 uint16 (0-65535) 存储，节省 4 倍内存 (相比 int64)
        print(f"Converting to numpy uint16...")
        total_tokens = len(token_ids)
        np_data = np.array(token_ids, dtype=np.uint16)
        
        # 3. 保存为二进制文件
        output_file = os.path.join(args.output_dir, f"{split_name}.bin")
        print(f"Saving {total_tokens} tokens to {output_file}...")
        np_data.tofile(output_file)
        print("Done.")

    process_split('train')
    process_split('validation')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-v1")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./data/wikitext_bin")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess_wikitext(args)