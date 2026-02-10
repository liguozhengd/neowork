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
import time
import logging
import pickle
import socket
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict
import random

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
        
        Args:
            client_id: 客户端ID
            train_data: 训练数据加载器
            test_data: 测试数据加载器（可选）
            config: 配置字典
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
        self.local_epochs = self.config.get('local_epochs', 5)
        self.local_lr = self.config.get('local_lr', 0.01)
        self.batch_size = self.config.get('batch_size', 32)
        
        # 通信设置
        self.server_host = self.config.get('server_host', '127.0.0.1')
        self.server_port = self.config.get('server_port', 5000)
        self.socket = None
        
        # 训练状态
        self.current_round = 0
        self.local_metrics = []
        
        # 隐私保护
        self.use_dp = self.config.get('use_differential_privacy', False)
        self.dp_sigma = None
        if self.use_dp:
            self._setup_differential_privacy()
        
        logger.info(f"Client {client_id} initialized on {self.device}")
    
    def _setup_differential_privacy(self):
        """设置差分隐私参数"""
        epsilon = self.config.get('epsilon', 1.0)
        delta = self.config.get('delta', 1e-5)
        clip_threshold = self.config.get('clip_threshold', 1.0)
        
        # 计算噪声标准差
        sensitivity = clip_threshold
        self.dp_sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
        self.clip_threshold = clip_threshold
        
        logger.info(f"DP setup: epsilon={epsilon}, sigma={self.dp_sigma:.4f}")
    
    def connect_to_server(self):
        """连接到服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.server_host, self.server_port))
            logger.info(f"Client {self.client_id} connected to server "
                       f"{self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def listen_for_commands(self):
        """监听服务器命令"""
        try:
            while True:
                # 接收服务器消息
                server_data = self._receive_from_server()
                
                if server_data['type'] == 'welcome':
                    self.client_id = server_data['client_id']
                    logger.info(f"Assigned client ID: {self.client_id}")
                
                elif server_data['type'] == 'model_update':
                    self.handle_model_update(server_data)
                
                elif server_data['type'] == 'training_complete':
                    logger.info("Training completed by server")
                    break
                
                else:
                    logger.warning(f"Unknown message type: {server_data['type']}")
        
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        except Exception as e:
            logger.error(f"Error in client {self.client_id}: {e}")
        finally:
            self.disconnect()
    
    def handle_model_update(self, server_data: dict):
        """处理服务器发送的模型更新"""
        self.current_round = server_data['round']
        logger.info(f"Round {self.current_round}: Received global model")
        
        # 1. 设置本地模型
        global_state = server_data['model_state']
        self.setup_local_model(global_state)
        
        # 2. 本地训练
        local_update, metrics = self.local_train()
        
        # 3. 发送更新到服务器
        self.send_update_to_server(local_update, metrics)
    
    def setup_local_model(self, global_state_dict: Dict):
        """
        基于全局模型设置本地模型
        
        Args:
            global_state_dict: 全局模型状态字典
        """
        # 创建模型实例
        if self.config['model']['name'] == 'SimpleNN':
            from models import SimpleNN
            model_class = SimpleNN
        
        self.local_model = model_class(**self.config['model']['params']).to(self.device)
        
        # 加载全局模型参数
        self.local_model.load_state_dict(global_state_dict)
        
        # 设置优化器
        optimizer_name = self.config.get('local_optimizer', 'sgd')
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.local_model.parameters(),
                lr=self.local_lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.local_model.parameters(),
                lr=self.local_lr,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def local_train(self) -> Tuple[Dict, Dict]:
        """
        本地训练
        
        Returns:
            update_dict: 本地模型更新（与初始模型的差异）
            metrics: 训练指标
        """
        if self.local_model is None:
            raise ValueError("Local model not initialized")
        
        logger.info(f"Client {self.client_id} starting local training "
                   f"({self.local_epochs} epochs)")
        
        # 保存初始模型状态
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        # 训练模式
        self.local_model.train()
        
        # 训练指标
        epoch_losses = []
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                
                # 后向传播
                loss.backward()
                
                # 梯度裁剪（用于差分隐私）
                if self.use_dp:
                    self._clip_gradients()
                
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                batch_count += 1
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 进度日志
                if batch_idx % 10 == 0:
                    logger.debug(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} "
                                f"Batch {batch_idx}/{len(self.train_data)} - Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
            
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} "
                       f"completed - Avg Loss: {avg_epoch_loss:.4f}")
        
        # 计算本地更新
        update_dict = self._compute_model_update(initial_state)
        
        # 添加差分隐私噪声
        if self.use_dp:
            update_dict = self._add_dp_noise(update_dict)
        
        # 计算指标
        avg_loss = np.mean(epoch_losses)
        accuracy = 100. * correct / total if total > 0 else 0
        
        metrics = {
            'client_id': self.client_id,
            'round': self.current_round,
            'loss': avg_loss,
            'accuracy': accuracy,
            'data_size': len(self.train_data.dataset),
            'batches': len(self.train_data)
        }
        
        # 本地评估（可选）
        if self.test_data is not None:
            test_metrics = self.local_evaluate()
            metrics.update(test_metrics)
        
        self.local_metrics.append(metrics)
        
        logger.info(f"Client {self.client_id} local training completed - "
                   f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return update_dict, metrics
    
    def _compute_model_update(self, initial_state: Dict) -> Dict:
        """
        计算模型更新（本地模型与初始模型的差异）
        
        Args:
            initial_state: 初始模型状态
            
        Returns:
            模型更新字典
        """
        update_dict = OrderedDict()
        final_state = self.local_model.state_dict()
        
        for key in final_state.keys():
            update_dict[key] = final_state[key] - initial_state[key]
        
        return update_dict
    
    def _clip_gradients(self, max_norm: Optional[float] = None):
        """梯度裁剪（用于差分隐私）"""
        if max_norm is None:
            max_norm = self.clip_threshold if hasattr(self, 'clip_threshold') else 1.0
        
        torch.nn.utils.clip_grad_norm_(
            self.local_model.parameters(),
            max_norm=max_norm
        )
    
    def _add_dp_noise(self, update_dict: Dict) -> Dict:
        """添加差分隐私噪声到模型更新"""
        noisy_dict = OrderedDict()
        
        for key, tensor in update_dict.items():
            # 计算噪声
            noise = torch.normal(
                mean=0,
                std=self.dp_sigma,
                size=tensor.shape
            ).to(self.device)
            
            # 添加到更新
            noisy_dict[key] = tensor + noise
        
        logger.debug(f"Added DP noise with sigma={self.dp_sigma:.4f}")
        return noisy_dict
    
    def local_evaluate(self) -> Dict:
        """本地模型评估"""
        if self.test_data is None:
            return {}
        
        self.local_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(self.test_data)
        accuracy = 100. * correct / total
        
        self.local_model.train()
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }
    
    def send_update_to_server(self, update_dict: Dict, metrics: Dict):
        """发送更新到服务器"""
        # 计算数据权重（通常为数据量）
        weight = len(self.train_data.dataset)
        
        # 准备数据
        client_data = {
            'type': 'client_update',
            'client_id': self.client_id,
            'round': self.current_round,
            'update': update_dict,
            'weight': weight,
            'metrics': metrics
        }
        
        # 发送到服务器
        try:
            self._send_to_server(client_data)
            logger.info(f"Client {self.client_id} sent update to server "
                       f"(weight: {weight})")
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
    
    def _send_to_server(self, data: dict):
        """发送数据到服务器"""
        serialized = pickle.dumps(data)
        self.socket.sendall(len(serialized).to_bytes(4, 'big'))
        self.socket.sendall(serialized)
    
    def _receive_from_server(self) -> dict:
        """从服务器接收数据"""
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
                raise ConnectionError("Incomplete data received")
            data += chunk
        
        return pickle.loads(data)
    
    def disconnect(self):
        """断开服务器连接"""
        if self.socket:
            self.socket.close()
            logger.info(f"Client {self.client_id} disconnected")
    
    def save_local_metrics(self, filename: str = None):
        """保存本地训练指标"""
        if filename is None:
            filename = f"client_{self.client_id}_metrics.json"
        
        with open(filename, 'w') as f:
            json.dump(self.local_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filename}")

class FedProxClient(FedClient):
    """FedProx客户端实现（支持近端项）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = self.config.get('fedprox_mu', 0.01)
    
    def local_train(self) -> Tuple[Dict, Dict]:
        """FedProx本地训练（添加近端项）"""
        if self.local_model is None:
            raise ValueError("Local model not initialized")
        
        # 保存初始模型状态（用于近端项计算）
        global_state = copy.deepcopy(self.local_model.state_dict())
        
        # 调用父类训练，但修改损失函数
        self._original_criterion = self.criterion
        
        # 创建FedProx损失函数
        def fedprox_loss(output, target, model):
            # 标准交叉熵损失
            ce_loss = self._original_criterion(output, target)
            
            # 近端项
            proximal_term = 0
            for param_name, param in model.named_parameters():
                proximal_term += torch.sum((param - global_state[param_name]) ** 2)
            
            return ce_loss + (self.mu / 2) * proximal_term
        
        # 临时替换损失函数
        self.criterion = lambda output, target: fedprox_loss(output, target, self.local_model)
        
        # 执行训练
        update_dict, metrics = super().local_train()
        
        # 恢复原始损失函数
        self.criterion = self._original_criterion
        
        # 添加近端项信息到更新
        update_dict['metadata'] = {
            'proximal': global_state
        }
        
        return update_dict, metrics

def create_client(client_id: int, 
                  train_data: DataLoader,
                  test_data: Optional[DataLoader] = None,
                  config_file: str = 'config.json') -> FedClient:
    """创建客户端实例"""
    # 加载配置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 根据配置选择客户端类型
    if config.get('algorithm') == 'fedprox':
        client = FedProxClient(client_id, train_data, test_data, config)
    else:
        client = FedClient(client_id, train_data, test_data, config)
    
    return client

def create_non_iid_data(dataset, num_clients: int, 
                       shards_per_client: int = 2) -> List[List[int]]:
    """
    创建非IID数据分布
    
    Args:
        dataset: 完整数据集
        num_clients: 客户端数量
        shards_per_client: 每个客户端的碎片数
        
    Returns:
        分配给每个客户端的数据索引列表
    """
    # 按标签排序
    targets = np.array([target for _, target in dataset])
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
        client_data.append(client_indices)
    
    return client_data

if __name__ == "__main__":
    # 示例使用
    import argparse
    from torchvision import datasets, transforms
    
    parser = argparse.ArgumentParser(description='联邦学习客户端')
    parser.add_argument('--client_id', type=int, required=True,
                       help='客户端ID')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 准备数据（示例：MNIST）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载完整数据集
    full_dataset = datasets.MN