#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import argparse
import logging
from types import SimpleNamespace

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 确保能找到parquet_reader模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parquet_reader import ParquetReader

def create_test_config(args):
    """创建用于测试的配置对象"""
    config = SimpleNamespace()
    
    # 创建data子配置
    config.data = SimpleNamespace()
    config.data.file_root = args.data_path
    config.data.train_path = args.data_path
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.workers
    config.data.normalize_images = not args.no_normalize
    config.data.shuffle = args.shuffle
    config.data.env_id = args.env_id
    
    return config

def show_batch_info(batch):
    """打印批次数据的详细信息"""
    logger.info(f"Batch keys: {list(batch.keys())}")
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            sample = v[0] if v else "empty list"
            logger.info(f"  {k}: type=list, length={len(v)}, example={sample}")
        else:
            logger.info(f"  {k}: type={type(v)}")

def show_sample_details(sample):
    """显示单个样本的详细信息"""
    logger.info("\nSample details:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.numel() > 10:
            logger.info(f"  {k}: type={type(v)}, shape={v.shape}")
        else:
            logger.info(f"  {k}: {v}")

def main(args):
    """主测试函数"""
    logger.info(f"Testing ParquetReader with data from: {args.data_path}")
    
    # 创建配置
    config = create_test_config(args)
    
    # 创建ParquetReader并获取dataloader
    reader = ParquetReader(config)
    dataloader = reader.read_parquet(split=args.split)
    
    logger.info(f"Dataset size: {len(dataloader.dataset)} frames")
    logger.info(f"Batch count: {len(dataloader)} batches")
    
    # 获取一个批次进行测试
    if len(dataloader) > 0:
        for batch in dataloader:
            # 打印批次信息
            show_batch_info(batch)
            
            # 如果有instruction字段，分析instruction的情况
            if 'instruction' in batch:
                instructions = batch['instruction']
                non_empty = sum(1 for instr in instructions if isinstance(instr, str) and instr.strip())
                logger.info(f"\nInstruction statistics:")
                logger.info(f"  Total: {len(instructions)}")
                logger.info(f"  Non-empty: {non_empty} ({non_empty/len(instructions)*100:.1f}%)")
                
                # 显示几个样例
                logger.info(f"  Examples:")
                for i in range(min(5, len(instructions))):
                    instr = instructions[i]
                    logger.info(f"    [{i}]: {instr}")
            
            # 显示一个完整样本
            if args.show_sample and len(batch) > 0:
                # 创建单个样本字典
                sample = {k: v[0] if isinstance(v, (torch.Tensor, list)) and len(v) > 0 else v for k, v in batch.items()}
                show_sample_details(sample)
            
            # 只处理一个批次
            break
    else:
        logger.error("No batches found in dataloader!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ParquetReader functionality")
    parser.add_argument("--data_path", type=str, default= './data/env_airsim_16/astar_data/high_average', help="Path to parquet data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--no_normalize", action="store_true", help="Disable image normalization")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--env_id", type=str, default="env_airsim_16", help="Environment ID to filter HF dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--show_sample", action="store_true", help="Show detailed sample information")
    args = parser.parse_args()
    
    main(args) 