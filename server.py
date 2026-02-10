"""
联邦学习服务器端实现
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import time
import logging
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Optional
import socket
import pickle
import threading
import select

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
        self.num_clients = config.get('num_clients', 10)
        self.num_rounds = config.get('num_rounds', 20)
        self.fraction = config.get('fraction', 0.3)  # 每轮选择的客户端比例
        self.aggregation_method = config.get('aggregation_method', 'fedavg')
        
        # 客户端管理
        self.client_updates = {}  # client_id -> update_dict
        self.client_weights = {}  # client_id -> data_weight
        self.client_metrics = defaultdict(list)  # 客户端性能指标
        
        # 训练状态
        self.current_round = 0
        self.best_accuracy = 0.0
        
        # 通信设置
        self.host = config.get('server_host', '127.0.0.1')
        self.port = config.get('server_port', 5000)
        self.socket = None
        
        # 可选：服务器端优化器（用于FedAdam等算法）
        self.server_optimizer = None
        if config.get('use_server_optimizer', False):
            self._setup_server_optimizer()
        
        logger.info(f"Server initialized on {self.device}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    def _setup_server_optimizer(self):
        """设置服务器端优化器"""
        if self.config.get('server_optimizer') == 'fedadam':
            from optimizers import FedAdamOptimizer
            self.server_optimizer = FedAdamOptimizer(
                self.global_model.state_dict(),
                lr=self.config.get('server_lr', 0.01),
                betas=self.config.get('betas', (0.9, 0.999)),
                tau=self.config.get('tau', 1e-3)
            )
    
    def start_server(self):
        """启动服务器监听客户端连接"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.num_clients)
        
        logger.info(f"Server started on {self.host}:{self.port}")
        logger.info(f"Waiting for {self.num_clients} clients to connect...")
        
        # 接受客户端连接
        self.connected_clients = {}
        for client_id in range(self.num_clients):
            client_socket, address = self.socket.accept()
            self.connected_clients[client_id] = {
                'socket': client_socket,
                'address': address,
                'status': 'connected'
            }
            logger.info(f"Client {client_id} connected from {address}")
            
            # 发送欢迎消息和配置
            welcome_msg = {
                'type': 'welcome',
                'client_id': client_id,
                'config': self.config
            }
            self._send_to_client(client_id, welcome_msg)
    
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
            self.aggregate()
            
            # 5. 评估全局模型
            if round_num % self.config.get('eval_every', 1) == 0:
                self.evaluate_global_model(round_num)
            
            # 6. 保存检查点
            if round_num % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(round_num)
        
        logger.info("Federated training completed!")
        self.finalize()
    
    def select_clients(self, round_num: int) -> List[int]:
        """
        选择参与本轮训练的客户端
        
        Args:
            round_num: 当前轮次
            
        Returns:
            选择的客户端ID列表
        """
        # 随机选择策略
        num_selected = max(1, int(self.num_clients * self.fraction))
        available_clients = list(range(self.num_clients))
        selected = np.random.choice(available_clients, num_selected, replace=False).tolist()
        
        # 可选：实现其他选择策略
        if self.config.get('client_selection') == 'power_of_choice':
            # 基于客户端性能的选择策略
            pass
        
        return selected
    
    def distribute_model(self) -> Dict:
        """分发全局模型到客户端"""
        # 创建模型的深拷贝
        model_state = copy.deepcopy(self.global_model.state_dict())
        return model_state
    
    def collect_client_updates(self, selected_clients: List[int]):
        """收集选中的客户端更新"""
        self.client_updates.clear()
        self.client_weights.clear()
        
        for client_id in selected_clients:
            # 接收客户端更新
            try:
                client_data = self._receive_from_client(client_id)
                
                if client_data['type'] == 'client_update':
                    update_dict = client_data['update']
                    weight = client_data['weight']
                    metrics = client_data.get('metrics', {})
                    
                    self.client_updates[client_id] = update_dict
                    self.client_weights[client_id] = weight
                    self.client_metrics[client_id].append(metrics)
                    
                    logger.info(f"Received update from client {client_id}, "
                               f"weight: {weight}, loss: {metrics.get('loss', 'N/A')}")
            
            except Exception as e:
                logger.error(f"Error receiving update from client {client_id}: {e}")
    
    def aggregate(self):
        """
        聚合客户端更新
        
        支持多种聚合算法：
        - fedavg: 联邦平均
        - weighted_avg: 加权平均
        - fedprox: FedProx算法
        - fednova: FedNova算法
        """
        if not self.client_updates:
            logger.warning("No client updates to aggregate")
            return
        
        logger.info(f"Aggregating updates from {len(self.client_updates)} clients "
                   f"using {self.aggregation_method}")
        
        # 获取全局模型状态
        global_dict = self.global_model.state_dict()
        
        if self.aggregation_method == 'fedavg':
            self._fedavg_aggregation(global_dict)
        
        elif self.aggregation_method == 'weighted_avg':
            self._weighted_avg_aggregation(global_dict)
        
        elif self.aggregation_method == 'fedprox':
            self._fedprox_aggregation(global_dict)
        
        elif self.aggregation_method == 'fednova':
            self._fednova_aggregation(global_dict)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # 应用服务器端优化器（如FedAdam）
        if self.server_optimizer is not None:
            global_dict = self.server_optimizer.step(
                list(self.client_updates.values()),
                list(self.client_weights.values()),
                global_dict
            )
        
        # 更新全局模型
        self.global_model.load_state_dict(global_dict)
    
    def _fedavg_aggregation(self, global_dict: Dict):
        """FedAvg聚合算法"""
        num_clients = len(self.client_updates)
        for key in global_dict.keys():
            # 初始化累加器
            global_dict[key] = torch.zeros_like(global_dict[key])
            
            # 累加所有客户端更新
            for client_id, update in self.client_updates.items():
                if key in update:
                    global_dict[key] += update[key]
            
            # 平均
            global_dict[key] /= num_clients
    
    def _weighted_avg_aggregation(self, global_dict: Dict):
        """加权平均聚合算法"""
        total_weight = sum(self.client_weights.values())
        
        for key in global_dict.keys():
            # 初始化累加器
            global_dict[key] = torch.zeros_like(global_dict[key])
            
            # 加权累加
            for client_id, update in self.client_updates.items():
                if key in update:
                    weight = self.client_weights[client_id] / total_weight
                    global_dict[key] += update[key] * weight
    
    def _fedprox_aggregation(self, global_dict: Dict):
        """FedProx聚合算法（需要客户端发送近端项）"""
        mu = self.config.get('fedprox_mu', 0.01)
        
        total_weight = sum(self.client_weights.values())
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            
            for client_id, update in self.client_updates.items():
                if key in update:
                    weight = self.client_weights[client_id] / total_weight
                    # FedProx考虑近端项
                    if 'proximal' in self.client_updates[client_id].get('metadata', {}):
                        prox_update = self.client_updates[client_id]['metadata']['proximal']
                        if key in prox_update:
                            update[key] = update[key] - mu * prox_update[key]
                    
                    global_dict[key] += update[key] * weight
    
    def _fednova_aggregation(self, global_dict: Dict):
        """FedNova聚合算法（处理不同的本地迭代次数）"""
        # 实现FedNova聚合逻辑
        pass
    
    def evaluate_global_model(self, round_num: int):
        """评估全局模型性能"""
        # 在实际应用中，这里应该在一个独立的测试集上评估
        logger.info(f"Evaluating global model at round {round_num}")
        
        # 示例：保存模型用于后续评估
        if self.config.get('test_loader') is not None:
            # 如果有测试数据加载器，进行评估
            test_loader = self.config['test_loader']
            self.global_model.eval()
            
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = nn.functional.cross_entropy(output, target, reduction='sum')
                    total_loss += loss.item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / total
            
            logger.info(f"Test Results - Loss: {avg_loss:.4f}, "
                       f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
            
            # 保存最佳模型
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_checkpoint(round_num, is_best=True)
        
        else:
            # 如果没有测试数据，只保存模型
            self.save_checkpoint(round_num)
    
    def save_checkpoint(self, round_num: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'round': round_num,
            'global_model_state': self.global_model.state_dict(),
            'config': self.config,
            'best_accuracy': self.best_accuracy,
            'client_metrics': dict(self.client_metrics)
        }
        
        if is_best:
            filename = f"checkpoints/best_model_round{round_num}.pth"
        else:
            filename = f"checkpoints/model_round{round_num}.pth"
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def _send_to_client(self, client_id: int, data: dict):
        """发送数据到指定客户端"""
        try:
            client_socket = self.connected_clients[client_id]['socket']
            serialized = pickle.dumps(data)
            client_socket.sendall(len(serialized).to_bytes(4, 'big'))
            client_socket.sendall(serialized)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
    
    def _receive_from_client(self, client_id: int) -> dict:
        """从指定客户端接收数据"""
        try:
            client_socket = self.connected_clients[client_id]['socket']
            
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
                    raise ConnectionError("Incomplete data received")
                data += chunk
            
            return pickle.loads(data)
        
        except Exception as e:
            logger.error(f"Error receiving from client {client_id}: {e}")
            raise
    
    def finalize(self):
        """结束训练，清理资源"""
        logger.info("Finalizing server...")
        
        # 通知所有客户端训练结束
        for client_id in self.connected_clients:
            try:
                self._send_to_client(client_id, {'type': 'training_complete'})
            except:
                pass
        
        # 关闭连接
        for client_info in self.connected_clients.values():
            client_info['socket'].close()
        
        if self.socket:
            self.socket.close()
        
        logger.info("Server finalized")

class PrivacyServer(FedServer):
    """支持隐私保护的服务器"""
    def __init__(self, model: nn.Module, config: dict):
        super().__init__(model, config)
        
        # 隐私保护参数
        self.epsilon = config.get('epsilon', 1.0)
        self.delta = config.get('delta', 1e-5)
        self.clip_threshold = config.get('clip_threshold', 1.0)
        
        logger.info(f"Privacy parameters: epsilon={self.epsilon}, "
                   f"delta={self.delta}, clip_threshold={self.clip_threshold}")
    
    def _secure_aggregate(self, global_dict: Dict):
        """安全聚合（添加差分隐私噪声）"""
        # 首先进行常规聚合
        self._weighted_avg_aggregation(global_dict)
        
        # 添加差分隐私噪声
        sensitivity = self.clip_threshold
        sigma = sensitivity * np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon
        
        for key in global_dict.keys():
            noise = torch.normal(mean=0, std=sigma, size=global_dict[key].shape)
            noise = noise.to(self.device)
            global_dict[key] += noise
        
        logger.info(f"Added DP noise with sigma={sigma:.6f}")
    
    def aggregate(self):
        """重写聚合方法以支持隐私保护"""
        if self.config.get('use_differential_privacy', False):
            global_dict = self.global_model.state_dict()
            self._secure_aggregate(global_dict)
            self.global_model.load_state_dict(global_dict)
        else:
            super().aggregate()

def create_server(config_file: str = 'config.json'):
    """创建服务器实例"""
    # 加载配置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model_config = config['model']
    if model_config['name'] == 'SimpleNN':
        from models import SimpleNN
        model = SimpleNN(**model_config['params'])
    
    # 创建服务器
    if config.get('use_privacy', False):
        server = PrivacyServer(model, config)
    else:
        server = FedServer(model, config)
    
    return server

if __name__ == "__main__":
    # 示例使用
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习服务器')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    # 创建并启动服务器
    server = create_server(args.config)
    server.port = args.port
    
    # 启动服务器
    server.start_server()
    
    # 开始联邦训练
    server.federated_training()