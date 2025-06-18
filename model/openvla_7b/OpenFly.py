import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class OpenFly(nn.Module):
    """
    OpenFly use siglip and dinov2 to process images, use llama's tokenizer to process text.
    Images will be processed by both siglip and dinov2, then the features will be concatenated and feed into a MLP Projector to map to the text embedding dimension.
    Text(Instruction) will be tokenized by llama's tokenizer, after this it's already in the text embedding dimension.
    Finally, the text and image features after MLP will be concatenated and feed into Llama 2 7b to generate the action.
    """
    def __init__(self, config):
        super(OpenFly, self).__init__()
        self.config = config