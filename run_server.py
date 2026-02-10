#!/usr/bin/env python3
"""
启动联邦学习服务器的脚本
"""
import argparse
import json
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server import create_server, FedServer
from utils import load_config, create_directories, set_seed

def main():
    parser = argparse.ArgumentParser(description='启动联邦学习服务器')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口')
    parser.add_argument('--clients', type=int, default=10,
                       help='客户端数量')
    parser.add_argument('--rounds', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--model', type=str, default='SimpleNN',
                       help='模型类型')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": args.port,
                "num_clients": args.clients,
                "fraction": 0.3
            },
            "training": {
                "num_rounds": args.rounds,
                "algorithm": "fedavg"
            },
            "model": {
                "name": args.model,
                "params": {
                    "input_size": 784,
                    "hidden_size": 128,
                    "num_classes": 10
                }
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": args.seed
        }
        # 保存默认配置
        with open('config_default.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("默认配置已保存到 config_default.json")
    
    # 更新命令行参数
    if args.port != 5000:
        config['server']['port'] = args.port
    if args.clients != 10:
        config['server']['num_clients'] = args.clients
    if args.rounds != 20:
        config['training']['num_rounds'] = args.rounds
    if args.model != 'SimpleNN':
        config['model']['name'] = args.model
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 创建目录
    create_directories(config)
    
    # 创建服务器
    print(f"创建联邦学习服务器...")
    print(f"端口: {config['server']['port']}")
    print(f"客户端数量: {config['server']['num_clients']}")
    print(f"训练轮数: {config['training']['num_rounds']}")
    print(f"模型: {config['model']['name']}")
    
    try:
        server = create_server(config)
        server.port = config['server']['port']
        
        # 启动服务器
        server.start_server()
        
        # 开始联邦训练
        server.federated_training()
        
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    except Exception as e:
        print(f"服务器错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'server' in locals():
            server.finalize()

if __name__ == "__main__":
    import torch
    main()