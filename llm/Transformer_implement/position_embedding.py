import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 位置编码PE(pos, 2i) = sin(pos/10000^{2i/d_model})
# 位置编码PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})

def PositionEncoding(seq_len, d_model, n = 10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model / 2)):
            denominator = np.power(n, 2 * i / d_model)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    
    return P
class PositionalEncoding(nn.Module):
    '''
        位置编码模块
    '''
    
    def __init__(self, args):
        super().__init__()
        
        # block size 是序列最大长度
        pe = torch.zeros(args.block_szie, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
    