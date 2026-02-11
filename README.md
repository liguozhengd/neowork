# 这里是lgz，本项目仅供挑战杯赛联邦学习与差分隐私项目使用。

## 运行说明

项目结构：
text
federated-learning/

├── server.py           # 服务器主文件

├── models.py           # 模型定义

├── utils.py            # 工具函数

├── config.json         # 配置文件

├── run_server.py       # 服务器启动脚本

├── requirements.txt    # 依赖包

├── checkpoints/        # 模型保存目录

├── logs/               # 日志目录

├── results/            # 结果目录

└── data/               # 数据目录（自动下载）

安装依赖：
bash
pip install -r requirements.txt
启动服务器（简化版）：
bash
# 使用默认配置
python run_server.py

# 指定参数
python run_server.py --port 5000 --clients 3 --rounds 5
启动客户端：
需要配合 client.py 文件（您已有的），在多个终端中启动：

bash
# 终端1
python run_client.py --client_id 0 --server_port 5000

# 终端2
python run_client.py --client_id 1 --server_port 5000

# 终端3
python run_client.py --client_id 2 --server_port 5000

### 1.1

 配置说明
服务器配置：修改config.json中的server部分

训练配置：修改config.json中的training部分

模型配置：修改config.json中的model部分

数据配置：修改config.json中的data部分
自定义模型
在models.py中添加新的模型类，然后在config.json中指定模型名称。

这个完整的框架支持：

多种聚合算法（FedAvg, FedProx等）

IID和非IID数据分布

差分隐私保护

多GPU训练

模型检查点和日志记录

可扩展的架构


### 1.2主要改进：
修复了配置加载问题：create_server 函数现在正确处理字典和文件路径

增强了错误处理：添加了更完善的异常处理

改进了日志系统：支持文件和命令行双重输出

简化了配置：提供了默认配置和命令行参数覆盖

增加了工具函数：数据划分、统计、可视化等

优化了通信：添加了超时处理和连接状态管理

### 1.3改进：
修复了utils.py 中的导入问题

修复 server.py 中的 create_server 函数

### 1.4修复的主要问题：
utils.py: 添加了 import torch.nn as nn，修复了 nn.Module 未定义的问题

server.py:

修复了 create_server 函数处理配置参数的问题

添加了更好的错误处理

修复了模型导入的逻辑

run_server.py:

修复了配置加载逻辑

修复了导入问题

所有文件: 统一了配置访问方式，使用 .get() 方法避免键不存在的问题