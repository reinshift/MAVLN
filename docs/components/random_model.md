# Random模型

Random模型是最简单的基线模型，模型随机生成动作，不考虑输入的视觉数据或指令。

## 模型架构

```python
class RandomAction:

    def __init__(self, config):
        self.config = config
        self.action_map = config.model.action_map
        self.num_agents = config.data.num_agents

    def forward(self):
        actions = {}
        for i in range(self.num_agents):
            actions[i] = random.randint(0, len(self.action_map) - 1)
        return actions
    
    def __call__(self):
        return self.forward()
```

## 使用方法

使用Random模型非常简单：

```python
from model.Random.Random import RandomAction
from utils.ConfigParser import load_config

# 加载配置
config = load_config("configs/common.yaml")

# 初始化模型
model = RandomAction(config)

# 生成随机动作
actions = model()
print(f"随机生成的动作: {actions}")
```