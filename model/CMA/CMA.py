import logging
from model.encoders import InstructionBertEncoder, InstructionEncoder, ResnetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import yaml
from utils.ConfigParser import Config

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """cross-modal attention to fuse the image and instruction"""
    def __init__(self, vision_dim, instruction_dim, hidden_dim):
        super(Attention, self).__init__()

        # projection
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.instruction_proj = nn.Linear(instruction_dim, hidden_dim)

        # attention
        self.query = nn.Linear(instruction_dim, hidden_dim) # instruction be mapped as query
        self.key = nn.Linear(vision_dim, hidden_dim) # image be mapped as key
        self.value = nn.Linear(vision_dim, hidden_dim) # image be mapped as value

        # output
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

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

        # calculate attention
        q = self.query(instr_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]
        k = self.key(vision_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]
        v = self.value(vision_feat).view(batch_size, 1, -1) # [batch_size, 1, hidden_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # [B, 1, 1]
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, 1, 1]
        attn_output = torch.matmul(attn_probs, v) # [B, 1, hidden_dim]
        attn_output = attn_output.view(batch_size, -1) # [B, hidden_dim]

        # residual connection & normalization
        output = self.norm1(instr_proj + attn_output)

        # ffn
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)

        return output

class CMA(nn.Module):
    """
    Baseline: Cross-Modal Attention (CMA) model
    This model is for single agent training, in multi-agent scenario, will use the same model for all agents.
    """
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
        else:
            raise ValueError(f"Invalid instruction encoder: {config.model.instruction_encoder}")

        self.resnet_encoder = ResnetEncoder(output_size=hidden_size)

        self.cross_attn = Attention(vision_dim=hidden_size, 
                                    instruction_dim=hidden_size, 
                                    hidden_dim=hidden_size)
        
        # action head
        num_actions = len(self.action_map)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_actions)
        )

        # image preprocess
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.config.model.output_para_num:
            logger.info(f"Number of parameters: {self._return_paras_count}")
        
        self.to(self.device)
    
    def forward(self, images, instructions):
        """
        Args:
            images: [batch_size, 3, H, W]
            instructions: [batch_size, seq_len]
        Returns:
            action_logits: [batch_size, num_actions]
        """
        vision_feats = self.resnet_encoder(images) # [batch_size, hidden_size]
        instr_feats = self.instruction_encoder(instructions) # [batch_size, hidden_size]
        
        fused_features = self.cross_attn(vision_feats, instr_feats) # [batch_size, hidden_size]

        action_logits = self.action_head(fused_features) # [batch_size, num_actions]

        return action_logits

    def take_action(self, image, instruction=None):
        """take action for single agent"""
        if not isinstance(image, Image.Image):
            img = Image.fromarray(image)
        else:
            img = image
        img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)

        if instruction is not None:
            if self.config.model.instruction_encoder == "bert":
                instructions_input = self.instruction_encoder.encode_text(instruction)
                instructions_input = {k: v.to(self.device) for k, v in instructions_input.items()}
            else:
                instructions_input = torch.zeros((1, 50), dtype=torch.long).to(self.device)
        else:
            logger.warning("No instruction provided, using empty instruction")
            instructions_input = torch.zeros((1, 50), dtype=torch.long).to(self.device)

        with torch.no_grad():
            action_logits = self.forward(img_tensor, instructions_input)
            action_id = torch.argmax(action_logits, dim=1).cpu().numpy()
            action_id = action_id.item()
            action_name = self.action_map[action_id]
        
        return action_name

    def take_actions(self, agent_images, instructions=None):
        """
        Args:
            agent_images: list: images of each agent, size: [num_agents, PIL.Image]
            instructions: list: instructions of each agent, size: [num_agents, str]
        Returns:
            action_dict: dict, key: agent_id, value: action_ID
        """
        action_dict = {}
        for agent_id, image, instruction in zip(range(len(agent_images)), agent_images, instructions):
            action_name = self.take_action(image, instruction)
            action_dict[agent_id] = action_name

        return action_dict

    @property
    def _return_paras_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Args:
            config_path: config file path
            
        Returns:
            CMA model instance
        """        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = Config(config_dict)
        return cls(config)
    
    def save_model(self, path, epoch=None, optimizer=None):

        def config_to_dict(config_obj):
            if not isinstance(config_obj, Config):
                return config_obj
            
            result = {}
 
            for k, v in config_obj.__dict__.items():
                if k == '_dict_props':
                    continue
                result[k] = config_to_dict(v)
            
            if hasattr(config_obj, '_dict_props'):
                for k, v in config_obj._dict_props.items():
                    result[k] = v
                    
            return result
        
        config_dict = config_to_dict(self.config)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': config_dict
        }
        
        if epoch is not None:
            save_dict['epoch'] = epoch
            
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load_model(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device)
        
        def dict_to_config(d):
            if not isinstance(d, dict):
                return d
                
            config = Config(d)
            return config
        
        config = dict_to_config(checkpoint['config'])
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model = model.to(device)
            
        logger.info(f"Model loaded from {path}")
        return model, checkpoint