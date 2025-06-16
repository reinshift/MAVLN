import logging
from model.encoders import InstructionBertEncoder, InstructionEncoder, ResnetEncoder, rnn_state_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import yaml
from utils.ConfigParser import Config

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, vision_dim, instruction_dim, hidden_dim):
        super(Attention, self).__init__()
        self.vision_dim = vision_dim
        self.instruction_dim = instruction_dim
        self.hidden_dim = hidden_dim
        
        
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
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        
        