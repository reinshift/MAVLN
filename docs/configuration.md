# 配置文件

这里包含了对`./config`下的配置文件(`.yaml`)、以及Airsim的配置文件(`.json`)介绍

## 配置文件结构

MAVLN配置文件主要包含以下几个部分：

1. **模型配置**：控制模型类型和参数
2. **训练配置**：控制模型训练参数
3. **数据配置**：控制数据加载和处理

## 配置选项详解

### 模型配置

| 选项 | 描述 | 示例值 |
|-----|-----|-------|
| `model.name` | 模型名称 | mavln, cma, seq2seq, etc |
| `model.mantis_path` | Mantis模型路径 | "./Mantis/mantis-model" |
| `model.maitis_type` | Mantis模型类型 | default, Idefics2 |
| `model.instruction_encoder` | 指令编码器类型 | bert, transformer |
| `model.action_map` | 动作映射字典 | {0: "stop", 1: "go straight", ...} |
| `model.continue_train` | 是否继续训练 | true, false |
| `model.pretrained_path` | 预训练模型路径 | "./checkpoints/cma_20250616_131806_epoch25.pth" |
| `model.output_para_num` | 是否输出参数数量 | true, false |

### 训练配置

| 选项 | 描述 | 示例值 |
|-----|-----|-------|
| `training.device` | 训练设备 | cuda, cpu |
| `training.seed` | 随机种子 | 42 |
| `training.lr` | 学习率 | 1e-5 |
| `training.epochs` | 训练轮数 | 50 |
| `training.gradient_clip` | 梯度裁剪值 | 1.0 |
| `training.optimizer` | 优化器类型 | adamw, adam |
| `training.if_warmup` | 是否使用预热 | true, false |
| `training.save_dir` | 模型保存目录 | "./checkpoints" |
| `training.save_interval` | 保存间隔 | 5 |
| `training.loss_threshold` | 早停损失阈值 | 0.005 |

### 数据配置

| 选项 | 描述 | 示例值 |
|-----|-----|-------|
| `data.env_id` | 环境ID | "env_airsim_16" |
| `data.file_root` | 数据根目录 | "./data" |
| `data.train_path` | 训练数据路径 | "./data/env_airsim_16/astar_data/high_average" |
| `data.val_path` | 验证数据路径 | "./data/env_airsim_16/astar_data/low_short" |
| `data.test_path` | 测试数据路径 | "./data/env_airsim_16/astar_data/high_average" |
| `data.num_workers` | 数据加载工作线程数 | 4 |
| `data.batch_size` | 批处理大小 | 32 |
| `data.num_agents` | 无人机数量 | 6 |
| `data.normalize_images` | 是否归一化图像 | true, false |
| `data.shuffle` | 是否打乱数据 | true, false |

## 不同模型的配置差异

MAVLN系统中的不同模型有其特定的配置选项，其中：

### MAVLN模型

MAVLN模型依赖于Mantis多模态大模型，需要以下关键配置：
- `model.mantis_path`：指定Mantis模型的路径
- `model.maitis_type`：指定使用的Mantis变体（default或Idefics2）

### CMA模型

CMA模型使用自定义的跨模态注意力机制，关键配置包括：
- `model.instruction_encoder`：指定指令编码器类型（bert或transformer）
- `model.continue_train`：是否从现有检查点继续训练
- `model.pretrained_path`：预训练模型路径（如果继续训练）

## 配置加载和解析

配置文件通过`utils/ConfigParser.py`中的函数进行加载：

```python
def load_config(config_path):
    """
    加载配置文件并返回Config对象
    
    参数:
        config_path: 配置文件路径
        
    返回:
        Config对象，可通过属性访问配置项
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    return config

class Config:
    """配置类，允许通过属性访问配置项"""
    def __init__(self, d):
        self._dict_props = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if all(isinstance(key, int) for key in v.keys()):
                    self._dict_props[k] = v
                else:
                    setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
```

## 使用示例

### 从配置文件加载模型

```python
from utils.ConfigParser import load_config
from model.mavln.mavln import MAVLN
from model.CMA.CMA import CMA

# 加载MAVLN模型配置
mavln_config = load_config("configs/common.yaml")
mavln_model = MAVLN(mavln_config)

# 加载CMA模型配置
cma_config = load_config("configs/cma.yaml")
cma_model = CMA(cma_config)
```

### 从命令行指定配置文件

```python
import argparse
from utils.ConfigParser import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/common.yaml", 
                   help="Path to config file")
args = parser.parse_args()

config = load_config(args.config)
```

## AirSim配置

除了MAVLN系统配置，AirSim本身也需要配置。AirSim配置文件通常位于`~/Documents/AirSim/settings.json`，我这里定义了6架无人机，示例如下（仿真器启动时如果只看到一架，那估计是模型重叠了，需要控制其中的无人机移动到不同位置）：

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "SpringArmChase",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "UAV_1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "Cameras": {
        "front_custom": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 144
            }
          ]
        }
      }
    },
    "UAV_2": {
      "VehicleType": "SimpleFlight",
      "X": 5, "Y": 0, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true
    },
    "UAV_3": {
      "VehicleType": "SimpleFlight",
      "X": 10, "Y": 0, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true
    },
    "UAV_4": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 5, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true
    },
    "UAV_5": {
      "VehicleType": "SimpleFlight",
      "X": 5, "Y": 5, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true
    },
    "UAV_6": {
      "VehicleType": "SimpleFlight",
      "X": 10, "Y": 5, "Z": -2,
      "EnableCollisions": true,
      "AllowAPIAlways": true
    }
  },
  "SubWindows": [
    {"WindowID": 0, "CameraName": "UAV_1", "ImageType": 0, "VehicleName": "UAV_1"},
    {"WindowID": 1, "CameraName": "UAV_2", "ImageType": 0, "VehicleName": "UAV_2"},
    {"WindowID": 2, "CameraName": "UAV_3", "ImageType": 0, "VehicleName": "UAV_3"}
  ],
  "OriginGeopoint": {
    "Latitude": 47.641468,
    "Longitude": -122.140165,
    "Altitude": 122
  },
  "TimeOfDay": {
    "Enabled": false,
    "StartDateTime": "2023-07-01 12:00:00",
    "CelestialClock": {
      "Speed": 1,
      "StartTime": 12
    }
  }
}
```
这里可以调整无人机的架数、无人机名称，以及各自所配备的传感器信息等，详情可见[官方文档](https://github.com/Microsoft/AirSim/blob/master/docs/settings.md)。