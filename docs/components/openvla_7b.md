# openvla-7b模型

[OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform)是一个VLN任务的数据生成平台，提供了本项目的数据源。该平台还微调了[openvla-7b](https://github.com/SHAILAB-IPEC/OpenFly-Platform#download-pretrained-weights)用于训练vln验证其数据的有效性。因此本项目将它微调的openfly-7b也作为对比模型。模型下载地址：[huggingface](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b)

## 代码结构

模型位于`model/openvla_7b/OpenFly.py`中尚未完成，后续要做的有：

- [x] 已经写了一个远程调用框架，但建议本地下载
- [ ] 下载模型权重到本地
- [ ] 模仿原项目写对应的IO pipeline