import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(InstructionEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.tokenizer = None
        
    def forward(self, instruction):
        """
        Args:
            instruction: [batch_size, seq_len]
            
        Returns:
            embedding: [batch_size, embedding_dim]
        """
        x = self.embedding(instruction)  # [batch_size, seq_len, embedding_dim]
        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        transformer_output = self.transformer_encoder(x)  # [seq_len, batch_size, embedding_dim]
        
        output = transformer_output[-1]  # [batch_size, embedding_dim]
        
        output = self.fc(output)  # [batch_size, embedding_dim]
        
        return output

    def init_tokenizer(self, vocab_path=None):
        """
        init tokenizer
        """
        if vocab_path:
            self.tokenizer = torch.load(vocab_path)
        else:
            self.tokenizer = lambda x: x.lower().split()

class InstructionBertEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", output_dim=512):
        super(InstructionBertEncoder, self).__init__()
        
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        self.bert_dim = self.bert_model.config.hidden_size
        
        if output_dim != self.bert_dim:
            self.dim_adapter = nn.Linear(self.bert_dim, output_dim)
        else:
            self.dim_adapter = nn.Identity()
            
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
    def forward(self, instruction):
        """
        Args:
            instruction: dict, contains input_ids, attention_mask, etc.
            
        Returns:
            embedding: [batch_size, output_dim]
        """
        outputs = self.bert_model(**instruction)
        
        pooled_output = outputs.pooler_output  # [batch_size, bert_dim]
        
        output = self.dim_adapter(pooled_output)  # [batch_size, output_dim]
        
        return output
        
    def encode_text(self, text_list):
        """
        encode text list to BERT input format
        
        Args:
            text_list: text string list
            
        Returns:
            encoded_input: dict, contains input_ids, attention_mask, etc.
        """
        encoded_input = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        return encoded_input