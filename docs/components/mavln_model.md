# MAVLN模型

MAVLN (Multi-Agent Vision-Language Navigation) 模型负责将多个无人机的视觉输入和自然语言指令转换为控制动作。该模型基于Mantis多模态大模型，具有处理多智能体输入并为每个智能体生成相应动作的能力。

## 模型架构

MAVLN模型基于Mantis大型视觉语言模型，主要功能是接收多个无人机的第一人称视角图像和导航指令，然后为每个无人机生成合适的动作。

MAVLN模型由以下主要部分组成：

1. **Mantis模型**：处理多模态输入（图像和文本）
2. **提示工程**：构建针对多智能体场景的提示模板
3. **动作解析器**：将模型生成的文本解析为具体的动作指令

## 代码结构

MAVLN模型的实现位于`model/mavln/mavln.py`文件中，主要类为`MAVLN`。

```python
class MAVLN(nn.Module):
    """
    Multi-Agent Vision-Language Navigation model based on Mantis.
    """
    def __init__(self, config):
        super(MAVLN, self).__init__()
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        self.num_agents = config.data.num_agents
        self.action_map = config.model.action_map
        
        # 加载Mantis模型
        model_path = config.model.mantis_path
        if config.model.maitis_type == "Idefics2":
            # 使用Idefics2版本的Mantis
            self.processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2")
            # ...初始化Idefics2版本的模型...
        elif config.model.maitis_type == "default":
            # 使用默认版本的Mantis
            self.processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
            self.model = LlavaForConditionalGeneration.from_pretrained(config.model.mantis_path,
                                                       device_map="auto",
                                                       torch_dtype=torch.bfloat16)
```

## 主要功能

### 1. 提示构建

`build_prompt`方法负责为多个无人机构建统一的提示，包括每个无人机的视觉信息和指令：

```python
def build_prompt(self, agent_images: List[Image.Image], instructions: List[str] = None) -> str:
    """
    构建多智能体导航提示 
    
    参数:
        agent_images: 每个智能体的图像列表 [agent1_image, agent2_image, ...]
        instructions: 每个智能体的指令列表
    
    返回:
        格式化的提示字符串
    """
    # 系统提示
    prompt = "You are commanding mutiple UAVs to navigate in an environment. "
    prompt += "Please base on their current views and instructions, determine the best action for each UAV.\n\n"
    
    # 添加每个智能体的视图和指令
    for i, agent_img_list in enumerate(agent_images):
        agent_name = f"Agent {i+1}"
        prompt += f"This is {agent_name}'s view: \n"
        
        # 图像将由处理器插入
        
        # 添加指令
        if instructions and i < len(instructions):
            prompt += f"\n{agent_name}'s instruction: {instructions[i]}\n\n"
        else:
            prompt += f"\n{agent_name} has no specific instruction.\n\n"
    
    # 添加可用动作
    prompt += "Actions available for each UAV:\n"
    for action_id, action_name in self.action_map.items():
        prompt += f"{action_id}: {action_name}\n"
    
    prompt += "\nPlease give the action number each UAV should take according their view and instruction:\n"
    prompt += "Format your response as 'Agent 1: [action_number], Agent 2: [action_number], ...' "
    
    return prompt
```

### 2. 模型前向传播

`forward`方法处理输入并返回模型输出：

```python
def forward(
        self,
        agent_images: List[List[Image.Image]],
        instructions: List[str] = None,
        return_logits: bool = False,
) -> Union[Dict[str, torch.Tensor], str]:
    """
    参数:
        agent_images: 每个智能体的图像列表 [agent1_image, agent2_image, ...]
        instructions: 每个智能体的指令列表
        return_logits: 是否返回logits(True)或生成的文本(False)
    
    返回:
        包含logits的模型输出或生成的文本
    """
    # 展平图像列表以进行处理
    all_images = []
    for agent_img_list in agent_images:
        all_images.extend(agent_img_list)
    
    # 构建提示
    prompt_text = self.build_prompt(agent_images, instructions)
    
    if self.config.model.maitis_type == "Idefics2":
        # 处理Idefics2类型模型
        messages = [{"role": "user", "content": [...]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=all_images, return_tensors="pt")
        # ...模型推理...
    elif self.config.model.maitis_type == "default":
        # 处理默认类型模型
        generation_kwargs = {"max_new_tokens": 1024, "num_beams": 1, "do_sample": False}
        response, _ = chat_mllava(prompt_text, all_images, self.model, self.processor, **generation_kwargs)
        return response
```

### 3. 响应生成

`generate_response`方法使用模型生成文本响应：

```python
def generate_response(self, inputs: Dict[str, torch.Tensor]) -> str:
    """
    返回: 生成的文本动作ID
    """
    generation_kwargs = {
        "max_new_tokens": self.model.generation_config.max_new_tokens,
        "num_beams": 1,
        "do_sample": True,
        "temperature": self.model.generation_config.temperature,
        "top_p": self.model.generation_config.top_p,
    }
    
    with torch.no_grad():
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
    
    response = self.processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return response
```

### 4. 动作解析

`parse_actions`方法将模型生成的文本解析为动作ID：

```python
def parse_actions(self, response: str) -> Dict[int, int]:
    """
    解析生成的响应，提取每个智能体的动作编号
    
    参数:
        response: 生成的文本响应
    
    返回:
        将智能体索引映射到动作编号的字典
    """
    actions = {}
    
    # 模糊匹配，防止模型不稳定输出非结构化文本
    import re
    action_patterns = re.findall(r"Agent\s+(\d+)\s*:\s*\[?(\d+)\]?", response)
    
    for agent_str, action_str in action_patterns:
        try:
            agent_idx = int(agent_str) - 1  # 转换为基于0的索引
            action_num = int(action_str)
            actions[agent_idx] = action_num
        except ValueError:
            continue
    
    # 检查是否找到了所有智能体的动作
    for i in range(self.num_agents):
        if i not in actions:
            actions[i] = 0  # 默认为停止
    
    return actions
```

## 配置加载

MAVLN模型支持从配置文件加载：

```python
@classmethod
def from_config_file(cls, config_path: str):
    """
    从配置文件加载模型 
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    return cls(config)
```

## 动作映射

MAVLN模型使用以下动作映射：

| 动作ID | 动作名称 | 描述 |
|:------:|:-------:|:----:|
| 0      | stop    | 停止移动 |
| 1      | go straight | 向前飞行 |
| 2      | turn left | 向左转向 |
| 3      | turn right | 向右转向 |
| 4      | go up | 上升 |
| 5      | go down | 下降 |
| 6      | move left | 向左移动 |
| 7      | move right | 向右移动 |

## 使用方法

### 初始化模型

```python
from model.mavln.mavln import MAVLN
from utils.ConfigParser import load_config

# 加载配置
config = load_config("configs/common.yaml")

# 初始化模型
model = MAVLN(config)

# 准备图像和指令（示例）
from PIL import Image
agent1_image = Image.open("path/to/agent1_image.jpg")
agent2_image = Image.open("path/to/agent2_image.jpg")
agent_images = [[agent1_image], [agent2_image]]
instructions = ["向前飞行直到看到红色建筑", "上升并向左转"]

# 模型推理
response = model(agent_images, instructions)
actions = model.parse_actions(response)
print(actions)  # {0: 1, 1: 4} - 智能体1:向前飞行, 智能体2:上升
```

## 训练和评估

目前直接用prompt控制结构化输出动作，没有训练微调过程，如果要训练可以在此基础上继续修改 