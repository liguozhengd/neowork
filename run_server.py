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

import torch
from server import create_server
from utils import setup_logging, create_directories, set_seed

def main():
    parser = argparse.ArgumentParser(description='启动联邦学习服务器')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--port', type=int, default=None,
                       help='服务器端口')
    parser.add_argument('--clients', type=int, default=None,
                       help='客户端数量')
    parser.add_argument('--rounds', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--model', type=str, default=None,
                       help='模型类型')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging('./logs')
    print(f"日志文件: {log_file}")
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        print(f"加载配置文件: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        from utils import get_default_config
        config = get_default_config()
        
        # 保存默认配置
        with open('config_default.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("默认配置已保存到 config_default.json")
    
    # 用命令行参数覆盖配置
    if args.port is not None:
        config['server']['port'] = args.port
    if args.clients is not None:
        config['server']['num_clients'] = args.clients
    if args.rounds is not None:
        config['training']['num_rounds'] = args.rounds
    if args.model is not None:
        config['model']['name'] = args.model
    if args.seed is not None:
        config['seed'] = args.seed
    
    # 设置随机种子
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # 创建目录
    create_directories(config)
    
    # 显示配置信息
    print("\n" + "="*60)
    print("联邦学习服务器配置")
    print("="*60)
    print(f"服务器地址: {config['server']['host']}:{config['server']['port']}")
    print(f"客户端数量: {config['server']['num_clients']}")
    print(f"训练轮数: {config['training']['num_rounds']}")
    print(f"每轮选择比例: {config['server']['fraction']}")
    print(f"聚合方法: {config['server']['aggregation_method']}")
    print(f"模型: {config['model']['name']}")
    print(f"设备: {config.get('device', 'cpu')}")
    print(f"随机种子: {seed}")
    print("="*60 + "\n")
    
    try:
        # 创建服务器
        print("正在创建服务器...")
        server = create_server(config)
        
        if server is None:
            print("服务器创建失败!")
            return
        
        # 启动服务器
        print("启动服务器监听...")
        connected_clients = server.start_server()
        
        if connected_clients == 0:
            print("没有客户端连接，服务器退出")
            return
        
        print(f"{connected_clients} 个客户端已连接，开始联邦训练...")
        
        # 开始联邦训练
        server.federated_training()
        
        print("联邦训练完成!")
        
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    except Exception as e:
        print(f"服务器错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'server' in locals():
            print("清理服务器资源...")
            server.finalize()
            print("服务器已关闭")

if __name__ == "__main__":
    main()