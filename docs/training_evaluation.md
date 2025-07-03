# 训练与评估

在`./train`中是训练相关代码，下面介绍`trainer.py`的各个方法以及使用（以CMA模型的训练为例）

## Trainer类

#### 1. 初始化

```python
def __init__(self, config):
    self.config = config
    self.device = config.training.device
    self.lr = float(config.training.lr)
    # ... 其他参数初始化
    
    # swanlab: to record training process
    self.swanlab_run = swanlab.init(
        project="train-VLN-model",
        experiment_name=self.config.model.name,
        # ... 配置参数
    )
```

初始化根据配置文件设置包括：
- 设备配置（CPU/GPU）
- 学习率和优化器配置
- 训练轮数和批处理大小
- 梯度裁剪阈值
- 模型保存配置
- 初始化实验记录工具（swanlab）

#### 2. 随机种子设置

```python
def setup_seed(self):
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)
    # ... 其他随机性控制
```

#### 3. 单轮训练

```python
def epoch_train(self, model, batch):
    images = batch['image'].to(self.device)
    # ... 处理指令和动作数据
    
    action_ids = torch.tensor(action_ids).to(self.device)
    
    # 根据模型类型选择训练流程
    if self.config.model.name == "cma":
        criterion = nn.CrossEntropyLoss()
        # ... 前向传播、损失计算和反向传播
        return loss.item()
    # ... 其他模型类型的处理
```

此方法处理单个批次的训练:
- 将图像和指令数据加载到指定设备
- 将动作标签转换为模型可用的格式
- 处理无效动作标签（自动纠正或使用默认值）
- 根据模型类型执行不同的训练步骤
- 记录损失和学习率
- 返回当前批次的损失值

#### 4. 主训练循环（单次）

```python
def train(self):
    # 数据加载
    data_reader = ParquetReader(self.config)
    train_loader = data_reader.read_parquet(split='train')
    
    # 模型初始化
    model = InitialModel(self.config)
    # ... 优化器和学习率调度器设置
    
    # 训练循环
    for epoch in range(self.epochs):
        # ... 进度条设置
        for batch_idx, batch in enumerate(train_loader):
            # ... 学习率调整
            loss = self.epoch_train(model, batch)
            # ... 进度记录
        
        # 模型保存
        if (epoch + 1) % self.save_interval == 0 or epoch + 1 == self.epochs:
            # ... 创建保存目录和文件名
            model.save_model(path=save_path, epoch=epoch+1, optimizer=self.optimizer)
```

主训练方法包含完整的训练流程：
1. 数据准备：使用`ParquetReader`加载训练数据
2. 模型初始化：创建模型实例并配置优化器
3. 学习率调度：设置学习率调度器和预热策略
4. 训练循环：迭代每个epoch和批次
5. 进度监控：使用tqdm显示训练进度和损失
6. 模型保存：定期保存模型检查点
7. 早停机制：当损失低于阈值时提前结束训练
这里其实写的不太好，应该写在模型类里的

#### 5. 评估方法（TODO）

```python
def epoch_evaluate(self):
    pass

def evaluate(self):
    pass
```

这些方法目前是占位符

## 训练流程

#### 配置训练环境

训练前需要准备合适的配置文件，配置文件在`./configs`目录下，包括：

- `common.yaml`: 通用配置，包含数据路径、训练参数等
- `cma.yaml`: CMA模型特定配置
- `seq2seq.yaml`: Seq2Seq模型特定配置
- `openfly.yaml`: OpenFly模型特定配置

#### 训练过程

CMA模型训练流程如下：

1. **初始化训练器**：`Trainer`类负责整个训练过程
2. **数据加载**：使用`ParquetReader`加载训练数据
3. **模型初始化**：通过配置文件初始化模型
4. **训练循环**：循环训练指定的epoch数
    - 每个batch进行前向传播、损失计算、反向传播和参数更新
    - 支持学习率预热和调度
    - 支持梯度裁剪
    - 使用swanlab记录训练过程
5. **模型保存**：定期保存模型检查点

#### CMA模型架构

CMA（Cross-Modal Attention）模型包含以下组件：

1. **图像编码器**：使用ResNet提取图像特征
2. **指令编码器**：根据配置使用BERT或Transformer编码文本指令
3. **跨模态注意力**：融合图像和指令特征
4. **动作头**：预测动作类型

## 训练命令

训练模型的命令示例：

```bash
# 训练CMA模型
python main.py --config configs/cma.yaml

# 继续训练（从检查点恢复）
python main.py --config configs/cma.yaml --continue_train True --pretrained_path checkpoints/cma_20250110_121212_epoch10.pth
```

## 配置参数说明

见[配置参数](./configuration.md)

## 训练监控与日志

训练过程使用swanlab进行监控，记录以下指标：

- 训练损失
- 学习率变化
- 训练时间

## 模型保存与加载

模型保存包含以下内容：

- 模型状态字典（`model_state_dict`）
- 配置信息（`config`）
- 当前轮数（`epoch`）
- 优化器状态（`optimizer_state_dict`）

模型加载示例：

```python
from model.CMA.CMA import CMA

# 加载模型
model, checkpoint = CMA.load_model(
    path="checkpoints/cma_20250110_121212_epoch10.pth",
    device="cuda:0"
)

# 使用模型推理
action = model.take_action(image, instruction)
```