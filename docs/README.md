# MAVLN - 多智能体视觉语言导航系统

> 基于多模态大模型Mantis的多智能体视觉语言导航系统

## 项目概述

MAVLN (Multi-Agent Vision-Language Navigation) 是一个多智能体视觉语言导航系统，利用AirSim仿真环境和多模态大模型Mantis，实现多无人机的视觉导航任务。系统通过将多个无人机的第一人称视角图像和自然语言指令输入到MAVLN模型，生成相应的控制动作，实现无人机在复杂环境中的自主导航。

## 为什么用Mantis作为视觉骨干网络

Mantis基于Llama3在多图输入任务的微调，对多图的理解能力较好，并且由于OpenVLA也是基于Llama2实现的，故在多智能体场景中考虑使用这个模型。目前停留在直接让其输出结构化动作并解析的阶段，测试过一段时间应该是比较稳定，同时如果想微调的话，代码里也保留了logits部分。

## 对比模型

目前包含以下几种模型：

- **Random**：随机动作模型，作为性能基线
- **MAVLN**：基于Mantis的多模态大模型，处理多个无人机的输入并生成动作
- **CMA**：采用交叉模态注意力机制，使用跨模态注意力机制
- **Seq2Seq**：序列到序列模型

## 主要组件

- **模型实现**：包含各种模型的定义和实现
- **数据处理**：处理Parquet格式的训练数据
- **训练系统**：负责模型训练、评估和参数管理
- **配置系统**：加载和解析YAML格式的配置文件

## 项目结构

```
MAVLN/
├── model/                     # 模型实现
│   ├── mavln/                 # MAVLN模型
│   ├── CMA/                   
│   ├── openvla_7b             # openvla-7b模型 (TODO)
│   ├── Random/                
│   ├── Seq2Seq/               
│   └── encoders/              # 共享编码器组件
├── Mantis/                    # 使用到的Mantis模型
├── docs/                      # 本开发文档
├── env/                       # airsim仿真环境
├── train/                     # 训练相关代码
├── utils/                     # 工具类，主要是解析数据用
├── configs/                   # 配置文件
├── data/                      # 数据集（.parquet）
├── scripts/                   # 测试用脚本，目前只有debug用代码，不是测试模型
├── results/                   # 上述测试的结果
├── airsim_bridge.py           # AirSim桥接服务器，用于启动打包好的airsim环境exe文件，传输指令控制其中的无人机
├── airsim_client.py           # AirSim客户端
└── main.py                    # 主程序入口
```

## 模型使用

在运行模型之前，需要下载Mantis模型以保证文件结构如下：

```
.
├── configs
├── Mantis 
│   ├── mantis-model # 在此处下载Mantis模型
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── model-00002-of-00004.safetensors
│   │   ├── model-00003-of-00004.safetensors
|   |   ├── ...
├── ...
```

> 在训练过程中使用[swanlab](https://swanlab.cn/)来跟踪实验日志。如果不想记录这些信息，可以注释掉`./train/train.py`中相关代码。