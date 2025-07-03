# 数据处理

模型需要处理多模态数据，包括图像数据、指令文本和动作标签。这里介绍要用的数据格式、处理流程和相关工具。

## 数据格式

### 训练数据格式

数据集来自[OpenFly](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)，以Parquet文件格式存储，每条记录包含以下字段：

- `env_id`: 环境ID，例如 "env_airsim_16"
- `traj_id`: 轨迹ID，格式为 "年-月-日_时-分-秒_xxx"，如 "2025-1-9_23-9-47_1237979969"
- `image_id`: 图像ID
- `frame_index`: 帧索引，从0开始
- `bytes`: 图像二进制数据
- `path`: 图像路径
- `element`/`pos`: 三维位置坐标，格式为 [x, y, z]
- `yaw`: 偏航角，用弧度表示
- `action_type`: 动作类型，包括 "go straight", "turn right", "turn left", "stop" 等
- `action_value`: 动作值，整数类型

每个Parquet文件代表一条完整轨迹，包含多个时间帧（通常为几十帧），用于训练序列模型。

### 数据目录结构

数据按以下结构组织：

```
data/
  └── env_airsim_16/
      └── astar_data/
          └── high_average/
              ├── 2025-1-9_23-9-47_1237979969.parquet
              ├── 2025-1-9_23-9-22_695258232.parquet
              └── ...
```

- `env_airsim_16`: 环境ID
- `astar_data`: 数据收集方法
- `high_average`: 难度级别
- 每个parquet文件: 一条完整轨迹

## 数据处理流程

### 数据加载和处理

项目使用`ParquetReader`类（位于`utils/parquet_reader.py`）来加载和处理数据：

1. **初始化**：设置数据路径、转换函数和批处理大小
2. **读取Parquet文件**：使用pyarrow库读取parquet文件
3. **提取指令**：从OpenFly数据集中提取对应的指令文本
4. **创建数据集**：将parquet文件封装为PyTorch数据集
5. **数据加载器**：创建DataLoader用于批量加载数据

### 图像预处理

图像数据处理包括：

- 调整大小至224x224
- 转换为张量
- 归一化（均值=[0.485, 0.456, 0.406]，标准差=[0.229, 0.224, 0.225]）

## 数据处理类和方法

### 1. ParquetReader 类

`ParquetReader` 类负责从 Parquet 文件中读取数据并创建 PyTorch 数据加载器。

#### 1.1 初始化方法

```python
def __init__(self, config):
    '''
    file path like: ./data/train/env_xxx/astar_data/high_long/xxx.parquet
    '''
    self.file_root = config.data.file_root
    self.config = config
    self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if getattr(config.data, 'normalize_images', True) else None
    self.batch_size = getattr(config.data, 'batch_size', 32)
    self.num_workers = getattr(config.data, 'num_workers', 4)
    self.shuffle = getattr(config.data, 'shuffle', True)
    self.env_id = getattr(config.data, 'env_id', 'env_airsim_16')
```

初始化方法接收配置对象，设置以下属性：
- `file_root`: 数据文件根目录
- `transform`: 图像预处理转换函数，包括调整大小、转换为张量和归一化
- `batch_size`: 批处理大小，默认为32
- `num_workers`: 数据加载的工作线程数，默认为4
- `shuffle`: 是否打乱数据，默认为True
- `env_id`: 环境ID，默认为'env_airsim_16'

#### 1.2 read_parquet 方法

```python
def read_parquet(self, split='train'):
    """
    读取 parquet 文件 & 创建 DataLoader
    
    参数:
        split: dataset split ('train', 'val', 'test')，目前只能用train
        
    返回:
        DataLoader
    """
    ...
```

这个方法负责读取 Parquet 文件并创建数据加载器：
1. 从 HuggingFace 加载 OpenFly 数据集获取指令文本（从上面看到parquet是没有包含指令的，必须回到hf里找）
2. 调用 `_extract_instruction` 方法提取指令（见下）
3. 根据配置确定数据目录
4. 创建 `TrajectoryDataset` 实例
5. 创建并返回 PyTorch DataLoader

#### 1.3 _extract_instruction 方法

```python
def _extract_instruction(self, dataset):
    '''
    Extract instruction from huggingface dataset
    '''
    instructions = {}
    matched = 0

    for item in tqdm.tqdm(dataset, desc="Extracting instructions"):
        if 'image_path' in item and self.env_id in item['image_path']:
            matched += 1
            path_parts = item['image_path'].split('/') # 'image_path' like 'env_airsim_16/astar_data/high_average/2025-1-9_16-8-22_1804289383'
            if len(path_parts) > 0:
                traj_id = path_parts[-1].split('.')[0] # '2025-1-9_16-8-22_1804289383'
                if 'gpt_instruction' in item and item['gpt_instruction']:
                    instructions[traj_id] = item['gpt_instruction'] # '2025-1-9_16-8-22_1804289383': 'go to the target'
    
    logger.info(f"Matched {matched} items")
    return instructions
```

从 HuggingFace 数据集中提取指令文本：
1. 遍历数据集中的每个样本
2. 检查图像路径是否包含目标环境 ID
3. 解析轨迹 ID
4. 从样本中提取 GPT 指令文本
5. 返回轨迹 ID 到指令的映射字典

### 2. TrajectoryDataset 类

`TrajectoryDataset` 类封装了轨迹数据处理逻辑。

#### 2.1 初始化

```python
def __init__(self, root_dirs, transform=None, instructions=None):
    self.root_dirs = root_dirs
    self.transform = transform
    self.file_paths = []
    self.metadata_df = None
    self.frames_df = None
    self.instructions = instructions or {}
    self._preload_metadata()
```

初始化方法接收以下参数：
- `root_dirs`: 数据目录列表
- `transform`: 图像转换函数
- `instructions`: 轨迹 ID 到指令的映射字典

初始化时会调用 `_preload_metadata()` 方法预加载所有 Parquet 文件的元数据。

#### 2.2 _preload_metadata 方法

```python
def _preload_metadata(self):
    '''
    预加载parquet的meta元素
    '''
    ...
    # 查找所有 Parquet 文件
    for root_dir in root_dirs:
    ... 
    # 提取每个文件的元数据
    data = []
    count_with_ins = 0
    for i, file_path in enumerate(tqdm.tqdm(self.file_paths, desc="Preloading metadata")):
        try:
            ...
    
    # 创建帧级索引数据框
    if len(self.metadata_df) > 0:
        frames_data = []
        ...
```

预加载所有 Parquet 文件的元数据：
1. 收集所有 Parquet 文件路径
2. 从每个文件中提取轨迹 ID 和帧数
3. 创建元数据数据框 `metadata_df`，包含轨迹级信息
4. 创建帧级数据框 `frames_df`，包含所有帧的索引信息

#### 2.3 __getitem__ 方法

```python
def __getitem__(self, idx):
    '''
    提取单个样本，用于后续batch化
    '''
    ...

        # 创建样本字典
        sample = {
            'image': image,
            'instruction': traj_instruction,
            'position': torch.tensor(row_data['pos'], dtype=torch.float),
            'yaw': torch.tensor(row_data['yaw'], dtype=torch.float),
            'action_type': row_data['action_type'],
            'action_value': torch.tensor(row_data['action_value'], dtype=torch.long),
            'traj_id': traj_id,
            'frame_index': frame_index
        }

        return sample

    ...
```

这个方法从数据集中获取单个样本：
1. 根据索引获取帧级记录
2. 读取对应的 Parquet 文件中的指定帧
3. 处理图像数据（解码和转换）
4. 获取对应的指令文本
5. 构建并返回样本字典，包含图像、指令、位置、偏航角和动作标签

#### 2.4 get_trajectory 方法

```python
def get_trajectory(self, traj_id):
    '''
    Get all frames from a specific trajectory
    
    Args:
        traj_id: trajectory ID
        
    Returns:
        List of samples from the trajectory in order
    '''
    traj_indices = self.frames_df[self.frames_df['traj_id'] == traj_id].sort_values('frame_index').index.tolist()
    return [self[idx] for idx in traj_indices]
```

这个方法获取特定轨迹的所有帧：
1. 从帧级数据框中筛选出指定轨迹的所有帧
2. 按帧索引排序
3. 使用 `__getitem__` 方法获取每一帧的数据
4. 返回包含所有帧的列表

## 使用示例

```python
# 配置文件中设置数据路径
config.data.file_root = "./data"
config.data.env_id = "env_airsim_16"

# 创建数据加载器
data_reader = ParquetReader(config)
train_loader = data_reader.read_parquet(split='train')

# 获取单个批次
for batch in train_loader:
    images = batch['image']
    instructions = batch['instruction']
    action_types = batch['action_type']
    # 模型训练...
    
# 获取完整轨迹
dataset = train_loader.dataset
trajectory = dataset.get_trajectory("2025-1-9_23-9-47_1237979969")
```