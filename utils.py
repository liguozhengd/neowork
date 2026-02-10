"""
工具函数
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn  # 修复：添加这行
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

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
        if directory:  # 确保目录路径不为空
            os.makedirs(directory, exist_ok=True)

def load_config(filename: str = 'config.json') -> Dict[str, Any]:
    """从文件加载配置"""
    if not os.path.exists(filename):
        logging.warning(f"Config file {filename} not found. Using default config.")
        return get_default_config()
    
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "num_clients": 10,
            "fraction": 0.3,
            "aggregation_method": "fedavg"
        },
        "client": {
            "local_epochs": 5,
            "local_lr": 0.01,
            "batch_size": 32,
            "local_optimizer": "sgd"
        },
        "model": {
            "name": "SimpleNN",
            "params": {
                "input_size": 784,
                "hidden_size": 128,
                "num_classes": 10
            }
        },
        "training": {
            "num_rounds": 20,
            "algorithm": "fedavg"
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            "results_dir": "./results"
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42
    }

def setup_logging(log_dir: str = "./logs", level=logging.INFO):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fedlearn_{timestamp}.log")
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def split_dataset(dataset, num_clients: int, non_iid: bool = True, 
                  shards_per_client: int = 2, seed: int = 42) -> List[List[int]]:
    """划分数据集给多个客户端"""
    set_seed(seed)
    
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
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            targets = np.array(dataset.labels)
        else:
            # 尝试通过索引获取
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
    except:
        return create_iid_split(dataset, num_clients)
    
    # 按标签排序
    sorted_indices = np.argsort(targets)
    
    # 划分碎片
    total_shards = num_clients * shards_per_client
    shard_size = len(dataset) // total_shards
    
    if shard_size == 0:
        shard_size = 1
        total_shards = min(len(dataset), total_shards)
    
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
        if client_shards:
            client_indices = np.concatenate(client_shards)
        else:
            client_indices = np.array([], dtype=np.int64)
        client_data.append(client_indices.tolist())
    
    return client_data

# 修复：compute_model_size 函数现在可以正常使用 nn.Module
def compute_model_size(model: nn.Module) -> float:
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())

# 默认导出
__all__ = [
    'set_seed', 'create_directories', 'load_config', 'get_default_config',
    'setup_logging', 'split_dataset', 'create_iid_split', 'create_non_iid_split',
    'compute_model_size', 'count_parameters'
]