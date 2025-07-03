# CMA 模型

CMA（Cross-Modal Attention）模型使用了交叉模态注意力机制，预期对齐文本和图片模态，用于处理图像输入和文本指令，输出导航动作。

## 模型架构

CMA模型由以下主要部分组成：

1. **图像编码器**：使用ResNet提取图像特征
2. **指令编码器**：支持BERT或自定义Transformer编码文本指令
3. **跨模态注意力层**：融合视觉和语言特征
4. **动作预测头**：生成导航动作

## 代码结构

CMA模型的实现位于`model/CMA/CMA.py`文件中，包含两个主要类：`Attention`和`CMA`。

```python
class CMA(nn.Module):
    def __init__(self, config, hidden_size=512):
        super(CMA, self).__init__()
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.action_map = config.model.action_map
        self.num_agents = config.data.num_agents
        self.hidden_size = hidden_size

        if config.model.instruction_encoder == "bert":
            self.instruction_encoder = InstructionBertEncoder(output_dim=hidden_size)
        elif config.model.instruction_encoder == "transformer":
            self.instruction_encoder = InstructionEncoder(output_dim=hidden_size)
        
        self.resnet_encoder = ResnetEncoder(output_size=hidden_size)
        self.cross_attn = Attention(vision_dim=hidden_size, 
                                   instruction_dim=hidden_size, 
                                   hidden_dim=hidden_size)
        
        # 动作预测头
        num_actions = len(self.action_map)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_actions)
        )
```

跨模态注意力机制通过`Attention`类实现：

```python
class Attention(nn.Module):
    """cross-modal attention to fuse the image and instruction"""
    def __init__(self, vision_dim, instruction_dim, hidden_dim):
        super(Attention, self).__init__()

        # 投影层
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.instruction_proj = nn.Linear(instruction_dim, hidden_dim)

        # 注意力机制
        self.query = nn.Linear(instruction_dim, hidden_dim) # 指令作为查询
        self.key = nn.Linear(vision_dim, hidden_dim) # 图像作为键
        self.value = nn.Linear(vision_dim, hidden_dim) # 图像作为值
        
        # 其他组件: 输出投影、归一化层和前馈网络
        # ...
```

## 主要功能

### 1. 前向传播

```python
def forward(self, images, instructions):
    """
    参数:
        图片: [batch_size, 3, H, W]
        指令: [batch_size, seq_len]
    返回:
        action_logits: [batch_size, num_actions]
    """
    vision_feats = self.resnet_encoder(images) # [batch_size, hidden_size]
    instr_feats = self.instruction_encoder(instructions) # [batch_size, hidden_size]
    
    fused_features = self.cross_attn(vision_feats, instr_feats) # [batch_size, hidden_size]
    action_logits = self.action_head(fused_features) # [batch_size, num_actions]

    return action_logits
```

- 使用图像编码器提取视觉特征
- 使用指令编码器提取文本特征
- 通过跨模态注意力机制融合特征
- 生成动作逻辑值

### 2. 单智能体动作生成

```python
def take_action(self, image, instruction=None):
    """take action for single agent"""
    if not isinstance(image, Image.Image):
        img = Image.fromarray(image)
    else:
        img = image
    img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)

    # 处理指令输入
    if instruction is not None:
        if self.config.model.instruction_encoder == "bert":
            instructions_input = self.instruction_encoder.encode_text(instruction)
            instructions_input = {k: v.to(self.device) for k, v in instructions_input.items()}
        else:
            instructions_input = torch.zeros((1, 50), dtype=torch.long).to(self.device)
    
    # 推理生成动作
    with torch.no_grad():
        action_logits = self.forward(img_tensor, instructions_input)
        action_id = torch.argmax(action_logits, dim=1).cpu().numpy().item()
        action_name = self.action_map[action_id]
    
    return action_name
```

此方法处理单个智能体的导航动作生成：
- 预处理输入图像
- 根据编码器类型处理指令
- 执行无梯度推理
- 返回预测的动作名称

### 3. 多智能体动作生成

```python
def take_actions(self, agent_images, instructions=None):
    """
    参数:
        每个agent的图像: list: images of each agent, size: [num_agents, PIL.Image]
        指令: list: instructions of each agent, size: [num_agents, str]
    返回:
        动作字典: dict, key: agent_id, value: action_ID
    """
    action_dict = {}
    for agent_id, image, instruction in zip(range(len(agent_images)), agent_images, instructions):
        action_name = self.take_action(image, instruction)
        action_dict[agent_id] = action_name

    return action_dict
```

此方法支持同时为多个智能体生成动作，用于和mavln对齐：
- 为每个智能体调用`take_action`方法
- 返回智能体ID到动作名称的映射字典

### 4. 模型保存与加载

```python
def save_model(self, path, epoch=None, optimizer=None):
    # 配置和模型状态保存逻辑
    save_dict = {
        'model_state_dict': self.state_dict(),
        'config': config_dict
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
        
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
    torch.save(save_dict, path)

@classmethod
def load_model(cls, path, device=None):
    # 模型加载逻辑
    checkpoint = torch.load(path, map_location=device)
    config = dict_to_config(checkpoint['config'])
    
    model = cls(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device:
        model = model.to(device)
        
    return model, checkpoint
```

这些方法实现了模型状态的保存和加载：
- 保存模型状态、配置、训练轮次和优化器状态
- 从检查点加载模型状态和配置

## 跨模态注意力机制

CMA模型的核心：

```python
def forward(self, vision_feat, instr_feat):
    """
    Args:
        vision_feat: [batch_size, vision_dim]
        instr_feat: [batch_size, instruction_dim]
    Returns:
        output: [batch_size, hidden_dim]
    """
    batch_size = vision_feat.size(0)
    instr_proj = self.instruction_proj(instr_feat)

    # 计算注意力
    q = self.query(instr_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]
    k = self.key(vision_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]
    v = self.value(vision_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v).view(batch_size, -1)

    # 残差连接和层归一化
    output = self.norm1(instr_proj + attn_output)
    
    # FFN和第二个残差连接
    ffn_output = self.ffn(output)
    output = self.norm2(output + ffn_output)

    return output
```

## 使用示例

### 初始化模型

```python
from model.CMA.CMA import CMA
from utils.ConfigParser import Config
import yaml

# 从配置文件初始化
with open("configs/cma.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = Config(config_dict)
model = CMA(config)

# 或者直接从配置文件路径初始化
model = CMA.from_config_file("configs/cma.yaml")
```

### 推理示例

```python
from PIL import Image

# 加载图像和准备指令
image = Image.open("path_to_image.jpg")
instruction = "向右转并直行到道路尽头"# 具体使用时要用英文

# 使用模型进行推理
action = model.take_action(image, instruction)
print(f"预测动作: {action}")
```

### 训练示例

```python
# 假设已加载数据和初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练一个批次
images = batch['image'].to(device)
instructions = model.instruction_encoder.encode_text(batch['instruction'])
instructions = {k: v.to(device) for k, v in instructions.items()}
action_ids = torch.tensor([model.action_map[a] for a in batch['action_type']]).to(device)

optimizer.zero_grad()
action_logits = model(images, instructions)
loss = criterion(action_logits, action_ids)
loss.backward()
optimizer.step()
```

## Seq2Seq model

这里还写了seq2seq对比模型，实现和cma有点像，只是文本编码采用的是rnn/lstm，故不再赘述。