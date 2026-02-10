"""
联邦学习客户端实现
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import json
import logging
import pickle
import socket
import os
from typing import Dict, Optional
from collections import OrderedDict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FedClient")

class FedClient:
    def __init__(self, 
                 client_id: int,
                 train_data: DataLoader,
                 test_data: Optional[DataLoader] = None,
                 config: Optional[dict] = None):
        """
        初始化联邦学习客户端
        """
        self.client_id = client_id
        self.config = config or {}
        
        # 设备设置
        self.device = torch.device(
            self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 数据
        self.train_data = train_data
        self.test_data = test_data
        
        # 本地模型和优化器
        self.local_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练参数
        self.local_epochs = self.config.get('client', {}).get('local_epochs', 5)
        self.local_lr = self.config.get('client', {}).get('local_lr', 0.01)
        
        # 通信设置
        self.server_host = self.config.get('server', {}).get('host', '127.0.0.1')
        self.server_port = self.config.get('server', {}).get('port', 5000)
        self.socket = None
        
        # 训练状态
        self.current_round = 0
        
        logger.info(f"Client {client_id} initialized")
    
    def connect_to_server(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            logger.info(f"Client {self.client_id} connected to server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def listen_for_commands(self):
        """监听服务器命令"""
        try:
            while True:
                # 接收服务器消息
                server_data = self._receive_from_server()
                
                if server_data['type'] == 'welcome':
                    logger.info(f"Client {self.client_id} received welcome")
                
                elif server_data['type'] == 'model_update':
                    self.handle_model_update(server_data)
                
                elif server_data['type'] == 'training_complete':
                    logger.info("Training completed")
                    break
                
                else:
                    logger.warning(f"Unknown message: {server_data['type']}")
        
        except KeyboardInterrupt:
            logger.info("Client interrupted")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.disconnect()
    
    def handle_model_update(self, server_data: dict):
        """处理服务器发送的模型更新"""
        self.current_round = server_data['round']
        logger.info(f"Round {self.current_round}: Received model")
        
        # 设置本地模型
        global_state = server_data['model_state']
        self.setup_local_model(global_state)
        
        # 本地训练
        local_update, metrics = self.local_train()
        
        # 发送更新到服务器
        self.send_update_to_server(local_update, metrics)
    
    def setup_local_model(self, global_state_dict: Dict):
        """基于全局模型设置本地模型"""
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'SimpleNN')
        
        try:
            from models import SimpleNN, CNNMnist, CNNCifar
            
            if model_name == 'SimpleNN':
                model_class = SimpleNN
            elif model_name == 'CNNMnist':
                model_class = CNNMnist
            elif model_name == 'CNNCifar':
                model_class = CNNCifar
            else:
                model_class = SimpleNN
        except ImportError:
            # 如果导入失败，创建简单模型
            class SimpleNN(nn.Module):
                def __init__(self, input_size=784, hidden_size=128, num_classes=10):
                    super().__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, hidden_size)
                    self.fc3 = nn.Linear(hidden_size, num_classes)
                
                def forward(self, x):
                    x = x.view(x.size(0), -1)
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            model_class = SimpleNN
        
        model_params = model_config.get('params', {})
        self.local_model = model_class(**model_params).to(self.device)
        self.local_model.load_state_dict(global_state_dict)
        
        # 设置优化器
        optimizer_name = self.config.get('client', {}).get('local_optimizer', 'sgd')
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.local_model.parameters(),
                lr=self.local_lr,
                momentum=self.config.get('client', {}).get('momentum', 0.9)
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.local_model.parameters(),
                lr=self.local_lr
            )
        else:
            self.optimizer = optim.SGD(self.local_model.parameters(), lr=self.local_lr)
    
    def local_train(self):
        """本地训练"""
        if self.local_model is None:
            raise ValueError("Local model not initialized")
        
        logger.info(f"Starting local training ({self.local_epochs} epochs)")
        
        # 保存初始模型状态
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        # 训练模式
        self.local_model.train()
        
        # 训练
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logger.info(f"Epoch {epoch+1}/{self.local_epochs} - Loss: {avg_loss:.4f}")
        
        # 计算本地更新
        update_dict = self._compute_model_update(initial_state)
        
        # 计算指标
        metrics = {
            'client_id': self.client_id,
            'round': self.current_round,
            'loss': avg_loss,
            'data_size': len(self.train_data.dataset)
        }
        
        return update_dict, metrics
    
    def _compute_model_update(self, initial_state: Dict) -> Dict:
        """计算模型更新"""
        update_dict = OrderedDict()
        final_state = self.local_model.state_dict()
        
        for key in final_state.keys():
            update_dict[key] = final_state[key] - initial_state[key]
        
        return update_dict
    
    def send_update_to_server(self, update_dict: Dict, metrics: Dict):
        """发送更新到服务器"""
        weight = len(self.train_data.dataset)
        
        client_data = {
            'type': 'client_update',
            'client_id': self.client_id,
            'round': self.current_round,
            'update': update_dict,
            'weight': weight,
            'metrics': metrics
        }
        
        try:
            self._send_to_server(client_data)
            logger.info(f"Sent update to server (weight: {weight})")
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
    
    def _send_to_server(self, data: dict):
        """发送数据到服务器"""
        serialized = pickle.dumps(data)
        self.socket.sendall(len(serialized).to_bytes(4, 'big'))
        self.socket.sendall(serialized)
    
    def _receive_from_server(self, timeout: float = 60.0) -> dict:
        """从服务器接收数据"""
        self.socket.settimeout(timeout)
        
        # 接收数据长度
        length_bytes = self.socket.recv(4)
        if not length_bytes:
            raise ConnectionError("Server disconnected")
        
        data_length = int.from_bytes(length_bytes, 'big')
        
        # 接收数据
        data = b''
        while len(data) < data_length:
            chunk = self.socket.recv(min(4096, data_length - len(data)))
            if not chunk:
                raise ConnectionError("Incomplete data")
            data += chunk
        
        return pickle.loads(data)
    
    def disconnect(self):
        """断开服务器连接"""
        if self.socket:
            self.socket.close()
            logger.info(f"Client {self.client_id} disconnected")

def create_client(client_id: int, 
                  train_data: DataLoader,
                  test_data: Optional[DataLoader] = None,
                  config: dict = None) -> FedClient:
    """创建客户端实例"""
    if config is None:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
    
    client = FedClient(client_id, train_data, test_data, config)
    return client

if __name__ == "__main__":
    # 测试客户端
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习客户端')
    parser.add_argument('--client_id', type=int, required=True,
                       help='客户端ID')
    parser.add_argument('--server_host', type=str, default='127.0.0.1',
                       help='服务器地址')
    parser.add_argument('--server_port', type=int, default=5000,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    print(f"启动客户端 {args.client_id}...")
    
    # 创建虚拟数据用于测试
    from torch.utils.data import TensorDataset, DataLoader
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建客户端
    client = FedClient(args.client_id, train_loader)
    client.server_host = args.server_host
    client.server_port = args.server_port
    
    if client.connect_to_server():
        try:
            client.listen_for_commands()
        except KeyboardInterrupt:
            print("客户端被中断")
    else:
        print("连接失败")