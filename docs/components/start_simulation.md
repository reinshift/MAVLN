# 启动脚本

启动脚本(`start_simulation.py`)用于管理AirSim桥接服务器和客户端的启动过程。该脚本简化了系统的启动流程，可以同时或分别启动桥接服务器和客户端，但由于是测试所以写的比较粗糙。

## 命令行参数

启动脚本支持以下命令行参数：

| 参数 | 描述 |
|-----|------|
| `--config` | 配置文件路径 |
| `--freq` | 客户端更新频率(Hz) (默认: 1.0) |
| `--wait-time` | 服务器启动后等待时间(秒) (默认: 2.0) |
| `--bridge-only` | 只启动桥接服务器 |
| `--client-only` | 只启动客户端 |
| `--detach` | 在后台运行进程 |
| `--instructions` | 固定指令文本文件路径 |
| `--log-level` | 日志级别 (默认: INFO) |

## 使用示例

### 完整启动

```bash
# 同时启动桥接服务器和客户端
python start_simulation.py --config configs/common.yaml --freq 2.0
```

### 只启动桥接服务器

```bash
# 只启动桥接服务器
python start_simulation.py --config configs/common.yaml --bridge-only
```

### 只启动客户端

```bash
# 只启动客户端(假设桥接服务器已经运行)
python start_simulation.py --config configs/common.yaml --client-only --freq 1.0
```

### 在后台运行

```bash
# 在后台运行，将输出重定向到日志文件
python start_simulation.py --config configs/common.yaml --detach > logs/simulation.log 2>&1
```

## 代码结构

启动脚本的主要功能在`start_simulation.py`文件中实现，主要包括以下几个部分：

### 参数解析

```python
def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="启动AirSim仿真环境和客户端")
    
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--freq", type=float, default=1.0, help="客户端更新频率(Hz)")
    parser.add_argument("--wait-time", type=float, default=2.0, help="服务器启动后等待时间(秒)")
    parser.add_argument("--bridge-only", action="store_true", help="只启动桥接服务器")
    parser.add_argument("--client-only", action="store_true", help="只启动客户端")
    parser.add_argument("--detach", action="store_true", help="在后台运行进程")
    parser.add_argument("--instructions", type=str, default=None, help="固定指令文本文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    return parser.parse_args()
```

### 启动桥接服务器

```python
def start_bridge_server(config, detach=False):
    """
    启动AirSim桥接服务器
    
    参数:
        config: 配置字典
        detach: 是否在后台运行
    
    返回:
        subprocess.Popen: 桥接服务器进程
    """
    bridge_cmd = ["python", "airsim_bridge.py"]
    
    # 添加配置选项
    if "server" in config:
        server_config = config["server"]
        if "host" in server_config:
            bridge_cmd.extend(["--host", server_config["host"]])
        if "port" in server_config:
            bridge_cmd.extend(["--port", str(server_config["port"])])
    
    # 启动进程
    logging.info("启动AirSim桥接服务器: %s", " ".join(bridge_cmd))
    
    if detach:
        # 在后台运行
        bridge_process = subprocess.Popen(
            bridge_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        # 在前台运行，输出到控制台
        bridge_process = subprocess.Popen(bridge_cmd)
    
    return bridge_process
```

### 启动客户端

```python
def start_client(config, freq, instructions=None, detach=False):
    """
    启动AirSim客户端
    
    参数:
        config: 配置字典
        freq: 更新频率(Hz)
        instructions: 固定指令文本
        detach: 是否在后台运行
    
    返回:
        subprocess.Popen: 客户端进程
    """
    client_cmd = ["python", "airsim_client.py", "--freq", str(freq)]
    
    # 添加配置文件
    if config:
        client_cmd.extend(["--config", config])
    
    # 添加指令文件
    if instructions:
        client_cmd.extend(["--instructions", instructions])
    
    # 启动进程
    logging.info("启动AirSim客户端: %s", " ".join(client_cmd))
    
    if detach:
        # 在后台运行
        client_process = subprocess.Popen(
            client_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        # 在前台运行，输出到控制台
        client_process = subprocess.Popen(client_cmd)
    
    return client_process
```

### 主函数

```python
def main():
    """
    主函数，处理启动流程
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=getattr(logging, args.log_level),
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 加载配置
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.error("加载配置文件失败: %s", str(e))
            return 1
    
    try:
        processes = []
        
        # 启动桥接服务器
        if not args.client_only:
            bridge_process = start_bridge_server(config, args.detach)
            processes.append(bridge_process)
            
            # 等待服务器启动
            logging.info("等待服务器启动 (%s 秒)...", args.wait_time)
            time.sleep(args.wait_time)
        
        # 启动客户端
        if not args.bridge_only:
            client_process = start_client(
                args.config, args.freq, args.instructions, args.detach
            )
            processes.append(client_process)
        
        # 等待进程完成
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        logging.info("接收到中断信号，正在终止进程...")
    finally:
        # 清理资源
        for process in processes:
            if process.poll() is None:  # 如果进程仍在运行
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    return 0
```
