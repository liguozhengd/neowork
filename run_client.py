#!/usr/bin/env python3
"""
启动联邦学习客户端的脚本
"""
import argparse
import json
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from client import create_client
from utils import load_config, set_seed, split_dataset

def prepare_data(config, client_id, num_clients):
    """准备客户端数据"""
    data_config = config.get('data', {})
    dataset_name = data_config.get('dataset', 'mnist')
    data_dir = data_config.get('data_dir', './data')
    batch_size = config.get('client', {}).get('batch_size', 32)
    
    # 数据转换
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载完整数据集
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 划分数据给客户端
    non_iid = data_config.get('non_iid', True)
    shards_per_client = data_config.get('shards_per_client', 2)
    
    client_indices = split_dataset(
        train_dataset, num_clients, non_iid, shards_per_client
    )
    
    # 获取当前客户端的数据
    if client_id < len(client_indices):
        indices = client_indices[client_id]
    else:
        indices = list(range(len(train_dataset)))
    
    # 创建数据加载器
    client_train_dataset = Subset(train_dataset, indices)
    client_test_dataset = test_dataset  # 使用完整的测试集
    
    train_loader = DataLoader(
        client_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('num_workers', 2)
    )
    
    test_loader = DataLoader(
        client_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 2)
    )
    
    print(f"客户端 {client_id}: 训练数据 {len(client_train_dataset)} 个样本")
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='启动联邦学习客户端')
    parser.add_argument('--client_id', type=int, required=True,
                       help='客户端ID')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--server_host', type=str, default='127.0.0.1',
                       help='服务器地址')
    parser.add_argument('--server_port', type=int, default=5000,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {
            "server": {
                "host": args.server_host,
                "port": args.server_port,
                "num_clients": 10
            },
            "client": {
                "local_epochs": 5,
                "local_lr": 0.01,
                "batch_size": 32
            },
            "model": {
                "name": "SimpleNN",
                "params": {
                    "input_size": 784,
                    "hidden_size": 128,
                    "num_classes": 10
                }
            },
            "data": {
                "dataset": "mnist",
                "data_dir": "./data",
                "non_iid": True
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42
        }
    
    # 更新命令行参数
    config['server']['host'] = args.server_host
    config['server']['port'] = args.server_port
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 准备数据
    num_clients = config['server']['num_clients']
    print(f"准备客户端 {args.client_id} 的数据...")
    train_loader, test_loader = prepare_data(config, args.client_id, num_clients)
    
    # 创建客户端
    print(f"创建客户端 {args.client_id}...")
    client = create_client(
        args.client_id,
        train_loader,
        test_loader,
        config
    )
    
    # 连接服务器
    print(f"连接到服务器 {args.server_host}:{args.server_port}...")
    if client.connect_to_server():
        try:
            # 监听服务器命令
            client.listen_for_commands()
        except KeyboardInterrupt:
            print(f"\n客户端 {args.client_id} 被用户中断")
        except Exception as e:
            print(f"客户端 {args.client_id} 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client.disconnect()
            # 保存客户端指标
            client.save_local_metrics(f"client_{args.client_id}_metrics.json")
    else:
        print(f"客户端 {args.client_id} 无法连接到服务器")

if __name__ == "__main__":
    main()