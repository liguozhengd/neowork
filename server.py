"""
联邦学习服务器端实现
"""
import torch
import torch.nn as nn
import numpy as np
import copy
import json
import logging
import os
import socket
import pickle
from collections import OrderedDict, defaultdict
from typing import List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FedServer")

class FedServer:
    def __init__(self, 
                 model: nn.Module,
                 config: dict):
        """
        初始化联邦学习服务器
        
        Args:
            model: 全局模型
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 全局模型
        self.global_model = model.to(self.device)
        
        # 训练参数
        self.num_clients = config.get('server', {}).get('num_clients', 10)
        self.num_rounds = config.get('training', {}).get('num_rounds', 20)
        self.fraction = config.get('server', {}).get('fraction', 0.3)
        self.aggregation_method = config.get('server', {}).get('aggregation_method', 'fedavg')
        
        # 客户端管理
        self.client_updates = {}  # client_id -> update_dict
        self.client_weights = {}  # client_id -> data_weight
        self.client_metrics = defaultdict(list)
        
        # 训练状态
        self.current_round = 0
        self.best_accuracy = 0.0
        
        # 通信设置
        self.host = config.get('server', {}).get('host', '127.0.0.1')
        self.port = config.get('server', {}).get('port', 5000)
        self.socket = None
        self.connected_clients = {}
        
        logger.info(f"Server initialized on {self.device}")
    
    def start_server(self):
        """启动服务器监听客户端连接"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.num_clients)
        
        logger.info(f"Server started on {self.host}:{self.port}")
        logger.info(f"Waiting for {self.num_clients} clients to connect...")
        
        # 接受客户端连接
        for client_id in range(self.num_clients):
            try:
                client_socket, address = self.socket.accept()
                self.connected_clients[client_id] = {
                    'socket': client_socket,
                    'address': address,
                    'status': 'connected'
                }
                logger.info(f"Client {client_id} connected from {address}")
                
                # 发送欢迎消息
                welcome_msg = {
                    'type': 'welcome',
                    'client_id': client_id,
                    'config': self.config
                }
                self._send_to_client(client_id, welcome_msg)
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
        
        return len(self.connected_clients)
    
    def federated_training(self):
        """执行联邦训练"""
        logger.info("Starting federated training...")
        
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_num + 1}/{self.num_rounds}")
            logger.info(f"{'='*60}")
            
            # 1. 选择客户端
            selected_clients = self.select_clients(round_num)
            if not selected_clients:
                logger.warning("No clients selected")
                continue
            
            logger.info(f"Selected clients: {selected_clients}")
            
            # 2. 分发全局模型
            global_state_dict = self.distribute_model()
            for client_id in selected_clients:
                self._send_to_client(client_id, {
                    'type': 'model_update',
                    'round': round_num,
                    'model_state': global_state_dict
                })
            
            # 3. 收集客户端更新
            self.collect_client_updates(selected_clients)
            
            # 4. 聚合更新
            if self.client_updates:
                self.aggregate()
            
            # 5. 保存检查点
            if round_num % self.config.get('training', {}).get('save_every', 5) == 0:
                self.save_checkpoint(round_num)
        
        logger.info("Federated training completed!")
        self.finalize()
    
    def select_clients(self, round_num: int) -> List[int]:
        """选择参与本轮训练的客户端"""
        num_selected = max(1, int(self.num_clients * self.fraction))
        available_clients = list(self.connected_clients.keys())
        
        if not available_clients:
            return []
        
        selected = np.random.choice(available_clients, 
                                   min(num_selected, len(available_clients)), 
                                   replace=False).tolist()
        return selected
    
    def distribute_model(self) -> Dict:
        """分发全局模型到客户端"""
        model_state = copy.deepcopy(self.global_model.state_dict())
        return model_state
    
    def collect_client_updates(self, selected_clients: List[int]):
        """收集选中的客户端更新"""
        self.client_updates.clear()
        self.client_weights.clear()
        
        for client_id in selected_clients:
            try:
                client_data = self._receive_from_client(client_id, timeout=60.0)
                
                if client_data['type'] == 'client_update':
                    update_dict = client_data['update']
                    weight = client_data['weight']
                    
                    self.client_updates[client_id] = update_dict
                    self.client_weights[client_id] = weight
                    
                    logger.info(f"Received update from client {client_id}, weight: {weight}")
            except socket.timeout:
                logger.error(f"Timeout from client {client_id}")
            except Exception as e:
                logger.error(f"Error from client {client_id}: {e}")
    
    def aggregate(self):
        """聚合客户端更新"""
        if not self.client_updates:
            return
        
        logger.info(f"Aggregating {len(self.client_updates)} client updates")
        
        global_dict = self.global_model.state_dict()
        
        if self.aggregation_method == 'fedavg':
            self._fedavg_aggregation(global_dict)
        elif self.aggregation_method == 'weighted_avg':
            self._weighted_avg_aggregation(global_dict)
        else:
            logger.warning(f"Unknown method: {self.aggregation_method}, using fedavg")
            self._fedavg_aggregation(global_dict)
        
        self.global_model.load_state_dict(global_dict)
    
    def _fedavg_aggregation(self, global_dict: Dict):
        """FedAvg聚合算法"""
        num_clients = len(self.client_updates)
        
        # 重置为0
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # 累加
        for client_id, update in self.client_updates.items():
            for key in global_dict.keys():
                if key in update:
                    global_dict[key] += update[key]
        
        # 平均
        for key in global_dict.keys():
            global_dict[key] /= num_clients
    
    def _weighted_avg_aggregation(self, global_dict: Dict):
        """加权平均聚合"""
        total_weight = sum(self.client_weights.values())
        
        if total_weight == 0:
            return
        
        # 重置为0
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # 加权累加
        for client_id, update in self.client_updates.items():
            weight = self.client_weights[client_id] / total_weight
            for key in global_dict.keys():
                if key in update:
                    global_dict[key] += update[key] * weight
    
    def save_checkpoint(self, round_num: int):
        """保存检查点"""
        checkpoint_dir = self.config.get('paths', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = f"{checkpoint_dir}/model_round_{round_num}.pth"
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def _send_to_client(self, client_id: int, data: dict):
        """发送数据到客户端"""
        try:
            client_socket = self.connected_clients[client_id]['socket']
            serialized = pickle.dumps(data)
            client_socket.sendall(len(serialized).to_bytes(4, 'big'))
            client_socket.sendall(serialized)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
    
    def _receive_from_client(self, client_id: int, timeout: float = 30.0) -> dict:
        """从客户端接收数据"""
        client_socket = self.connected_clients[client_id]['socket']
        client_socket.settimeout(timeout)
        
        try:
            # 接收数据长度
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                raise ConnectionError("Client disconnected")
            
            data_length = int.from_bytes(length_bytes, 'big')
            
            # 接收数据
            data = b''
            while len(data) < data_length:
                chunk = client_socket.recv(min(4096, data_length - len(data)))
                if not chunk:
                    raise ConnectionError("Incomplete data")
                data += chunk
            
            return pickle.loads(data)
        except Exception as e:
            raise ConnectionError(f"Error receiving: {e}")
    
    def finalize(self):
        """结束训练，清理资源"""
        logger.info("Finalizing server...")
        
        # 通知客户端训练结束
        for client_id in self.connected_clients:
            try:
                self._send_to_client(client_id, {'type': 'training_complete'})
            except:
                pass
        
        # 关闭连接
        for client_info in self.connected_clients.values():
            try:
                client_info['socket'].close()
            except:
                pass
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        logger.info("Server finalized")

# 修复：create_server 函数正确处理配置
def create_server(config: dict = None):
    """
    创建服务器实例
    
    Args:
        config: 配置字典，如果为None则从默认文件加载
    """
    if config is None:
        # 尝试从默认配置文件加载
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            from utils import get_default_config
            config = get_default_config()
    
    # 确保config是字典
    elif isinstance(config, str):
        # 如果传递的是文件路径字符串
        try:
            with open(config, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config} not found")
            return None
    
    # 创建模型
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'SimpleNN')
    
    try:
        # 尝试导入models模块
        from models import SimpleNN, CNNMnist, CNNCifar
        
        if model_name == 'SimpleNN':
            model_params = model_config.get('params', {})
            model = SimpleNN(**model_params)
        elif model_name == 'CNNMnist':
            model_params = model_config.get('params', {})
            model = CNNMnist(**model_params)
        elif model_name == 'CNNCifar':
            model_params = model_config.get('params', {})
            model = CNNCifar(**model_params)
        else:
            logger.warning(f"Unknown model: {model_name}, using SimpleNN")
            model = SimpleNN()
    except ImportError:
        # 如果导入失败，创建简单模型
        logger.warning("Cannot import models module, creating SimpleNN")
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
        
        model_params = model_config.get('params', {})
        model = SimpleNN(**model_params)
    
    # 创建服务器
    server = FedServer(model, config)
    return server

if __name__ == "__main__":
    # 示例使用
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习服务器')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--port', type=int, default=None,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        from utils import get_default_config
        config = get_default_config()
    
    if args.port is not None:
        config['server']['port'] = args.port
    
    server = create_server(config)
    if server:
        server.start_server()
        server.federated_training()
    else:
        logger.error("Failed to create server")