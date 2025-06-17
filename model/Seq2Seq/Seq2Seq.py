import logging
from model.encoders import InstructionBertEncoder, InstructionEncoder, ResnetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from utils.ConfigParser import Config

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """
    Different from CMA, Seq2Seq uses a normal attention mechanism to process concatenated image and instruction features.
    """
    def __init__(self, vision_dim, instruction_dim, hidden_dim):
        super(Attention, self).__init__()
        self.vision_dim = vision_dim
        self.instruction_dim = instruction_dim
        self.hidden_dim = hidden_dim
        input_dim = vision_dim + instruction_dim + 1 # 1 for action placeholder

        # projection
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        
        # output projection
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        
        # layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Attention mechanism for fusing features
        
        Args:
            x: concatenated features [batch_size, input_dim]
            
        Returns:
            out: attended features [batch_size, hidden_dim]
        """
        # Project to query, key, value
        q = self.W_q(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = self.W_k(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        v = self.W_v(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [batch_size, 1, 1]
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attn_probs, v)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection
        out = self.W_out(context)
        
        # Residual connection and layer normalization
        out = self.layer_norm(out)
        
        return out
        
class Seq2Seq(nn.Module):
    """
    Baseline: Seq2Seq model
    an LSTM-based architecture with an attention mechanism
    each step t:
        input:
            current image o_t       __
            previous action a_{t-1} __|-> hidden state h_t
            language feature \bar{x}_t
        output:
            next action prediction logits a_t
    
    encoders & decoders:
        image encoder: Resnet152
        language encoder: h_i = LSTM_{enc}(x_i, h_{i-1})
        action encoder: concat with image feature to form a single vector q_t
        decoder: h_t = LSTM_{dec}(q_t, h_{t-1})
        
    prediction:
        c_t = f(h_t, \bar{h}), \bar{h} = {h_1, ..., h_L} (instruction context)
        \widehat{h}_t = tanh(W_c[c_t; h_t])
        a_t = softmax(\widehat{h}_t)

    """
    def __init__(self, config, hidden_size=512):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.action_map = config.model.action_map
        self.num_agents = config.data.num_agents
        self.hidden_size = hidden_size

        # instruction encoder
        if config.model.instruction_encoder == "bert":
            self.instruction_encoder = InstructionBertEncoder(output_dim=hidden_size)
        elif config.model.instruction_encoder == "transformer":
            self.instruction_encoder = InstructionEncoder(output_dim=hidden_size)
        else:
            raise ValueError(f"Invalid instruction encoder: {config.model.instruction_encoder}")

        # image encoder
        self.resnet_encoder = ResnetEncoder(output_size=hidden_size)
        
        # LSTM encoder for instructions
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size*2 + 1,  # image features + instruction context + previous action
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # attention
        self._attn = Attention(vision_dim=hidden_size, 
                               instruction_dim=hidden_size,
                               hidden_dim=hidden_size)
        
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # normlayer
        self.norm_layer = nn.LayerNorm(hidden_size)

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
        
        # Hidden state for LSTM
        self.hidden_state = None
        self.cell_state = None
        
        # Previous action placeholder
        self.prev_action = None

        if self.config.model.output_para_num:
            logger.info(f"Number of parameters: {self._return_paras_count}")
    
    def forward(self, images, instructions, history_action=None):
        """
        Forward pass of Seq2Seq model
        
        Args:
            images: image tensor [batch_size, C, H, W]
            instructions: instruction tensor or dict
            history_action: previous action embedding [batch_size, 1]
            
        Returns:
            action_logits: predicted action logits [batch_size, num_actions]
        """
        batch_size = images.shape[0]
        
        # Extract features
        vision_feats = self.resnet_encoder(images)  # [batch_size, hidden_size]
        
        # Process instructions
        if self.config.model.instruction_encoder == "bert":
            # For BERT encoder, instructions is already a dict
            instr_feats = self.instruction_encoder(instructions)  # [batch_size, hidden_size]
        else:
            # For other encoders, process accordingly
            instr_feats = self.instruction_encoder(instructions)  # [batch_size, hidden_size]
        
        # If no history action provided, use -1
        if history_action is None:
            history_action = torch.ones((batch_size, 1), device=self.device) * -1
        
        # Concat features
        combined_input = torch.cat([vision_feats, instr_feats, history_action], dim=1)  # [batch_size, 2*hidden_size + 1]
        
        # Process through attention
        attn_output = self._attn(combined_input)  # [batch_size, hidden_size]
        
        # Process through LSTM decoder if hidden state exists
        if self.hidden_state is not None and self.cell_state is not None:
            # Reshape for LSTM input [batch_size, 1, feature_dim]
            lstm_input = combined_input.unsqueeze(1)
            
            # LSTM forward pass
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm_decoder(
                lstm_input, 
                (self.hidden_state, self.cell_state)
            )
            
            # Extract output
            lstm_out = lstm_out.squeeze(1)  # [batch_size, hidden_size]
            
            # Combine LSTM output with attention output
            fused_features = self.norm_layer(attn_output + lstm_out)
        else:
            # Initialize hidden states if not available
            self.hidden_state = torch.zeros((1, batch_size, self.hidden_size), device=self.device)
            self.cell_state = torch.zeros((1, batch_size, self.hidden_size), device=self.device)
            
            # Use attention output directly for first step
            fused_features = attn_output
        
        # Feed-forward network
        ffn_output = self.ffn(fused_features)
        fused_features = self.norm_layer(fused_features + ffn_output)
        
        # Action prediction
        action_logits = self.action_head(fused_features)
        
        return action_logits

    def take_action(self, image, instruction=None):
        """
        Predict a single action for a single image and instruction
        
        Args:
            image: PIL image or numpy array
            instruction: text instruction
            
        Returns:
            action_id: predicted action ID
            action_name: predicted action name
        """
        # Convert image to tensor
        if not isinstance(image, Image.Image):
            img = Image.fromarray(image)
        else:
            img = image
        img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
        
        # Process instruction
        if instruction is not None:
            if self.config.model.instruction_encoder == "bert":
                instructions_input = self.instruction_encoder.encode_text([instruction])
                instructions_input = {k: v.to(self.device) for k, v in instructions_input.items()}
            else:
                # Simplified handling for demonstration
                instructions_input = torch.zeros((1, 50), dtype=torch.long).to(self.device)
        else:
            # Handle empty instruction
            if self.config.model.instruction_encoder == "bert":
                instructions_input = self.instruction_encoder.encode_text([""])
                instructions_input = {k: v.to(self.device) for k, v in instructions_input.items()}
            else:
                instructions_input = torch.zeros((1, 50), dtype=torch.long).to(self.device)
        
        # Use previous action if available
        history_action = self.prev_action if self.prev_action is not None else torch.zeros((1, 1), device=self.device)
        
        # Get prediction
        with torch.no_grad():
            action_logits = self.forward(img_tensor, instructions_input, history_action)
            action_id = torch.argmax(action_logits, dim=1).cpu().numpy().item()
            action_name = self.action_map[action_id]
        
        # Store current action as previous for next step
        self.prev_action = torch.tensor([[action_id]], dtype=torch.float, device=self.device)
        
        return action_id, action_name

    def take_actions(self, agent_images, instructions=None):
        """
        Predict actions for multiple agents
        
        Args:
            agent_images: list of agent images
            instructions: list of instructions
            
        Returns:
            actions_dict: dictionary mapping agent indices to action IDs
        """
        actions_dict = {}
        
        for i, image in enumerate(agent_images):
            instruction = instructions[i] if instructions and i < len(instructions) else None
            action_id, _ = self.take_action(image, instruction)
            actions_dict[i] = action_id
            
        return actions_dict
    
    def reset_states(self):
        """Reset LSTM states and previous action"""
        self.hidden_state = None
        self.cell_state = None
        self.prev_action = None
        
    def save_model(self, path, epoch=None, optimizer=None):
        """Save model parameters and training state"""
        # Convert config to serializable dict
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
        """Load model from saved file"""
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

    @property
    def _return_paras_count(self):
        return sum(p.numel() for p in self.parameters())