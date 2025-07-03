# AirSim客户端

AirSim客户端(`airsim_client.py`)用于连接AirSim桥接服务器，获取无人机第一人称视角图像，与设置的指令一同馈入模型输出动作，并将动作指令解析以及发送回桥接服务器执行。

## 功能

- 连接到AirSim桥接服务器
- 获取无人机第一人称视角图像
- 将图像和指令输入到MAVLN模型
- 处理MAVLN模型的输出并转换为动作指令
- 将动作指令发送到桥接服务器执行

## 类和方法

### AirSimClient类

`AirSimClient`负责与桥接服务器通信并处理模型的输入输出。

```python
class AirSimClient:
    def __init__(self, uav_id=1, server_host="localhost", server_port=8888, 
                 config_path=None, freq=1.0):
        """
        初始化AirSim客户端
        
        参数:
            uav_id: 无人机ID
            server_host: 服务器主机名
            server_port: 服务器端口
            config_path: 配置文件路径
            freq: 更新频率(Hz)
        """
        pass
```

#### 主要方法

##### connect_to_server()

```python
def connect_to_server(self):
    """
    连接到AirSim桥接服务器
    
    返回:
        bool: 连接成功返回True，否则返回False
    """
```

创建与AirSim桥接服务器的TCP连接。

##### init_uav()

```python
def init_uav(self):
    """
    初始化无人机
    
    返回:
        bool: 初始化成功返回True，否则返回False
    """
```

发送初始化请求到桥接服务器，初始化指定ID的无人机。

##### get_image()

```python
def get_image(self):
    """
    获取无人机的图像
    
    返回:
        PIL.Image: 无人机视角图像，失败返回None
    """
```

从桥接服务器获取无人机的第一人称视角图像。

##### execute_action()

```python
def execute_action(self, action):
    """
    执行无人机动作
    
    参数:
        action: 要执行的动作ID
    
    返回:
        bool: 执行成功返回True，否则返回False
    """
```

向桥接服务器发送动作执行请求。

##### set_position()

```python
def set_position(self, position):
    """
    设置无人机位置
    
    参数:
        position: 位置坐标 [x, y, z, roll, pitch, yaw]
    
    返回:
        bool: 设置成功返回True，否则返回False
    """
```

向桥接服务器发送设置位置请求。

##### _send_request()

```python
def _send_request(self, data):
    """
    发送请求到服务器
    
    参数:
        data: 请求数据
    
    返回:
        dict: 服务器响应数据
    """
```

向桥接服务器发送请求并等待响应。

##### run()

```python
def run(self):
    """
    运行客户端主循环
    """
```

客户端主循环，定期获取图像，处理模型，执行动作。

##### load_model()

```python
def load_model(self):
    """
    加载MAVLN模型
    
    返回:
        object: 加载的模型实例
    """
```

加载和初始化MAVLN模型。

##### process_instruction()

```python
def process_instruction(self, image, instruction):
    """
    处理指令和图像，生成动作
    
    参数:
        image: 无人机视角图像
        instruction: 指令文本
    
    返回:
        int: 生成的动作ID
    """
```

将图像和指令输入MAVLN模型，生成动作ID。

##### stop()

```python
def stop(self):
    """
    停止客户端
    """
```

停止客户端，关闭所有连接。

## 命令行参数

AirSim客户端支持以下命令行参数：

| 参数 | 描述 |
|-----|------|
| `--uav-id` | 无人机ID (默认: 1) |
| `--server-host` | 服务器主机名 (默认: localhost) |
| `--server-port` | 服务器端口 (默认: 8888) |
| `--config` | 配置文件路径 |
| `--freq` | 更新频率(Hz) (默认: 1.0) |
| `--instruction` | 固定指令文本 |

## 模型调用

### 模型加载

客户端使用`utils.InitialModel`加载指定的模型：

```python
def load_model(self):
    self.model_manager = ModelManager(self.config_path)
    self.model = self.model_manager.load_model()
    return self.model
```

### 输入预处理

客户端使用以下步骤处理图像和指令：

1. 将Base64编码的图像数据解码为PIL.Image
2. 调整图像大小为模型所需尺寸
3. 将RGB图像转换为模型所需格式
4. 对指令文本进行tokenization

### 输出后处理

模型输出转换为动作ID：

```python
def process_instruction(self, image, instruction):
    # 预处理图像
    preprocessed_image = self._preprocess_image(image)
    
    # 预处理指令
    preprocessed_instruction = self._preprocess_instruction(instruction)
    
    # 模型推理
    outputs = self.model(preprocessed_image, preprocessed_instruction)
    
    # 获取动作ID
    action_id = outputs["action_id"]
    
    return action_id
```

## 使用示例

```bash
# 启动带有默认设置的客户端
python airsim_client.py

# 指定无人机ID和服务器主机
python airsim_client.py --uav-id 2 --server-host 192.168.1.100

# 指定配置文件和更新频率
python airsim_client.py --config configs/common.yaml --freq 2.0

# 提供固定指令
python airsim_client.py --instruction "向前飞行，然后在红色建筑物前降落"
```

## 多客户端操作（未测试）

客户端初步设计成支持多客户端同时连接到桥接服务器，每个客户端控制不同的无人机：

```bash
# 在不同终端启动多个客户端，分别控制不同无人机
terminal1$ python airsim_client.py --uav-id 1
terminal2$ python airsim_client.py --uav-id 2
terminal3$ python airsim_client.py --uav-id 3
```

## 与模型的接口

AirSim客户端与模型之间的接口如下：

```python
# 输入接口
inputs = {
    "image": preprocessed_image,  # 形状: [batch_size, channels, height, width]
    "instruction": preprocessed_instruction  # 形状: [batch_size, seq_length]
}

# 输出接口
outputs = {
    "action_id": action_id,  # 形状: [batch_size]
    "action_logits": action_logits,  # 形状: [batch_size, num_actions]
    "attention_weights": attention_weights  # 可视化注意力权重
}
``` 