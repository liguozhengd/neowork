# neowork
这是lgz,本仓库用作挑战杯赛的联邦学习与差分隐私项目
使用说明
1. 创建项目结构
text
federated-learning/
├── server.py           # 服务器端代码
├── client.py           # 客户端端代码
├── models.py           # 模型定义
├── utils.py            # 工具函数
├── config.json         # 配置文件
├── run_server.py       # 启动服务器脚本
├── run_client.py       # 启动客户端脚本
├── requirements.txt    # 依赖包
├── checkpoints/        # 模型检查点目录
├── logs/               # 日志目录
├── results/            # 结果目录
└── data/               # 数据目录
2. 安装依赖
bash
pip install -r requirements.txt
3. 启动服务器
bash
# 使用默认配置
python run_server.py

# 指定配置文件和参数
python run_server.py --config config.json --port 5000 --clients 10 --rounds 20
4. 启动客户端（需要开多个终端）
bash
# 终端1：客户端0
python run_client.py --client_id 0 --server_port 5000

# 终端2：客户端1  
python run_client.py --client_id 1 --server_port 5000

# 终端3：客户端2
python run_client.py --client_id 2 --server_port 5000

# ... 以此类推，启动多个客户端
5. 配置说明
服务器配置：修改config.json中的server部分

训练配置：修改config.json中的training部分

模型配置：修改config.json中的model部分

数据配置：修改config.json中的data部分

6. 自定义模型
在models.py中添加新的模型类，然后在config.json中指定模型名称。

这个完整的框架支持：

多种聚合算法（FedAvg, FedProx等）

IID和非IID数据分布

差分隐私保护

多GPU训练

模型检查点和日志记录

可扩展的架构
