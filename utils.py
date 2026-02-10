"""
工具函数
"""
import os
import json
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, List, Tuple

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config: Dict[str, Any]):
    """创建必要的目录"""
    paths = config.get('paths', {})
    
    directories = [
        paths.get('checkpoint_dir', './checkpoints'),
        paths.get('log_dir', './logs'),
        paths.get('results_dir', './results'),
        config.get('data', {}).get('data_dir', './data')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def save_config(config: Dict[str, Any], filename: str = 'config_saved.json'):
    """保存配置到文件"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logging.info(f"Config saved to {filename}")

def load_config(filename: str = 'config.json') -> Dict[str, Any]:
    """从文件加载配置"""
    if not os.path.exists(filename):
        logging.warning(f"Config file {filename} not found. Using default config.")
        return get_default_config()
    
    with open(filename, 'r') as f:
        config = json.load(f)
    
    return config

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "num_clients": 10,
            "fraction": 0.3
        },
        "training": {
            "num_rounds": 20,
            "local_epochs": 5
        },
        "model": {
            "name": "SimpleNN",
            "params": {
                "input_size": 784,
                "hidden_size": 128,
                "num_classes": 10
            }
        }
    }

def log_metrics(metrics: Dict[str, Any], round_num: int, logger: logging.Logger):
    """记录训练指标"""
    logger.info(f"Round {round_num} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

def save_checkpoint(state: Dict, filename: str):
    """保存检查点"""
    torch.save(state, filename)
    logging.info(f"Checkpoint saved: {filename}")

def load_checkpoint(filename: str, device: str = 'cpu'):
    """加载检查点"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint {filename} not found")
    
    checkpoint = torch.load(filename, map_location=device)
    logging.info(f"Checkpoint loaded from {filename}")
    return checkpoint

def split_dataset(dataset, num_clients: int, non_iid: bool = True, 
                  shards_per_client: int = 2) -> List[List[int]]:
    """划分数据集给多个客户端"""
    if non_iid:
        return create_non_iid_split(dataset, num_clients, shards_per_client)
    else:
        return create_iid_split(dataset, num_clients)

def create_iid_split(dataset, num_clients: int) -> List[List[int]]:
    """创建IID数据划分"""
    data_len = len(dataset)
    indices = list(range(data_len))
    random.shuffle(indices)
    
    split_size = data_len // num_clients
    splits = []
    
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_clients - 1 else data_len
        splits.append(indices[start_idx:end_idx])
    
    return splits

def create_non_iid_split(dataset, num_clients: int, 
                         shards_per_client: int = 2) -> List[List[int]]:
    """创建非IID数据划分（基于标签）"""
    # 获取数据标签
    try:
        targets = np.array([target for _, target in dataset])
    except:
        # 如果无法直接获取target，尝试其他方法
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # 按标签排序
    sorted_indices = np.argsort(targets)
    
    # 划分碎片
    total_shards = num_clients * shards_per_client
    shard_size = len(dataset) // total_shards
    
    shards = []
    for i in range(total_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < total_shards - 1 else len(dataset)
        shards.append(sorted_indices[start_idx:end_idx])
    
    # 随机分配碎片给客户端
    random.shuffle(shards)
    
    client_data = []
    for i in range(num_clients):
        client_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
        client_indices = np.concatenate(client_shards)
        client_data.append(client_indices.tolist())
    
    return client_data