import torch 
from torch import nn 
import math 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout:float) -> None:
        super().__init__(PositionalEncoding)
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zero(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Aplly the sin to even positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6)-> None:
        super().__init__(LayerNormalization)
        self.eps = eps
        self.alphs = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias - nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias 
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float)-> None:
        super().__init__(FeedForwardBlock)
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__(MultiHeadAttentionBlock)
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0, "d_model is not divisible by h"
    
        self.d_k = d_model // h
        self.w_q = nn.linear(d_model, d_model) #Wq
        self.w_k = nn.linear(d_model, d_model) #Wk
        self.w_v =nn.Linear(d_model, d_model)  #Wv

        self.w_o = nn.iinear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = 
