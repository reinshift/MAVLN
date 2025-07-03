# AirSim桥接服务器

AirSim桥接服务器(`airsim_bridge.py`)用于启动和管理AirSim仿真环境（项目路径./env下），以及处理客户端请求。它作为服务器端，接收来自客户端的命令，控制AirSim中的无人机，并将图像数据传回客户端。

## 功能

- 启动和管理AirSim仿真环境
- 为多个无人机创建并管理控制器
- 提供TCP服务器接收客户端连接
- 处理客户端请求(获取图像、执行动作等)
- 控制AirSim中的无人机执行动作

## 类和方法

### AirSimBridge类

`AirSimBridge`是主要类，负责与AirSim交互并管理客户端连接。

```python
class AirSimBridge:
    def __init__(self, host="localhost", port=8888, sim_path=None, config_path=None):
        """
        初始化AirSim桥接服务器
        
        参数:
            host: 服务器主机名
            port: 服务器端口
            sim_path: 模拟器路径
            config_path: 配置文件路径
        """
        ...
```

#### 主要方法

##### start_simulator()

```python
def start_simulator(self):
    """
    启动AirSim模拟器
    
    返回:
        bool: 启动成功返回True，否则返回False
    """
```

启动AirSim仿真环境，并等待初始化完成。

##### connect_to_airsim()

```python
def connect_to_airsim(self):
    """
    连接到AirSim客户端API
    
    返回:
        bool: 连接成功返回True，否则返回False
    """
```

创建与AirSim的API连接。

##### start_server()

```python
def start_server(self):
    """
    启动TCP服务器以接收客户端连接
    
    返回:
        bool: 启动成功返回True，否则返回False
    """
```

启动TCP服务器监听客户端连接请求。

##### _handle_connections()

```python
def _handle_connections(self):
    """
    处理客户端连接的内部方法
    """
```

接受客户端连接并为每个连接创建处理线程。

##### _handle_client()

```python
def _handle_client(self, client_socket, addr):
    """
    处理单个客户端连接
    
    参数:
        client_socket: 客户端socket连接
        addr: 客户端地址
    """
```

处理来自单个客户端的请求，包括初始化、获取图像和执行动作等。

##### _process_client_message()

```python
def _process_client_message(self, uav_id, data, client_socket):
    """
    处理来自客户端的消息
    
    参数:
        uav_id: 无人机ID
        data: 消息数据
        client_socket: 客户端socket连接
    """
```

解析和处理客户端的消息请求。

##### _init_uav_controller()

```python
def _init_uav_controller(self, uav_id):
    """
    初始化无人机控制器
    
    参数:
        uav_id: 无人机ID
    
    返回:
        bool: 初始化成功返回True，否则返回False
    """
```

在AirSim中为特定ID的无人机创建控制器。

##### get_uav_image()

```python
def get_uav_image(self, uav_id):
    """
    获取无人机的图像
    
    参数:
        uav_id: 无人机ID
    
    返回:
        PIL.Image: 无人机视角图像，失败返回None
    """
```

从AirSim获取指定无人机的第一人称视角图像。

##### execute_uav_action()

```python
def execute_uav_action(self, uav_id, action):
    """
    执行无人机动作
    
    参数:
        uav_id: 无人机ID
        action: 要执行的动作ID
    
    返回:
        bool: 执行成功返回True，否则返回False
    """
```

控制AirSim中的无人机执行指定动作。

##### start()

```python
def start(self):
    """
    启动AirSim桥接服务器
    
    返回:
        bool: 启动成功返回True，否则返回False
    """
```

启动桥接服务器，包括启动模拟器和TCP服务器。

##### stop()

```python
def stop(self):
    """
    停止AirSim桥接服务器
    """
```

停止桥接服务器，关闭所有连接并终止模拟器。

## 动作映射

AirSim桥接服务器使用以下动作映射将动作ID映射到AirSim中的具体控制命令：

| 动作ID | 动作名称 | 实现方式 |
|--------|---------|---------|
| 0      | stop    | moveByVelocityAsync(0, 0, 0, 1) |
| 1      | go straight | moveByVelocityAsync(1, 0, 0, 1) |
| 2      | turn left | rotateByYawRateAsync(-30, 1) |
| 3      | turn right | rotateByYawRateAsync(30, 1) |
| 4      | go up | moveByVelocityAsync(0, 0, -1, 1) |
| 5      | go down | moveByVelocityAsync(0, 0, 1, 1) |
| 6      | move left | moveByVelocityAsync(0, -1, 0, 1) |
| 7      | move right | moveByVelocityAsync(0, 1, 0, 1) |

## 命令行参数

AirSim桥接服务器支持以下命令行参数：

| 参数 | 描述 |
|-----|------|
| `--host` | 服务器主机名 (默认: localhost) |
| `--port` | 服务器端口 (默认: 8888) |
| `--sim-path` | AirSim模拟器路径 |
| `--config` | 配置文件路径 |

## 使用示例

```bash
# 启动带有默认设置的桥接服务器
python airsim_bridge.py

# 指定主机和端口
python airsim_bridge.py --host 0.0.0.0 --port 9000

# 指定模拟器路径
python airsim_bridge.py --sim-path /path/to/AirSim/simulator
```

### 1. 初始化请求

客户端发送:
```json
{
  "type": "init",
  "uav_id": 1
}
```

服务器响应:
```json
{
  "status": "ok",
  "message": "UAV 1 initialized"
}
```

### 2. 获取图像请求

客户端发送:
```json
{
  "type": "get_image"
}
```

服务器响应:
```json
{
  "status": "ok",
  "type": "image",
  "image": "base64编码的图像数据..."
}
```

### 3. 执行动作请求

客户端发送:
```json
{
  "type": "execute_action",
  "action": 1
}
```

服务器响应:
```json
{
  "status": "ok",
  "message": "execute action 1 successfully"
}
```

## 代码实现细节

桥接服务器使用Python的`socket`库实现TCP服务器，使用`threading`库实现多线程处理客户端请求，使用`airsim`库与AirSim仿真环境交互。 