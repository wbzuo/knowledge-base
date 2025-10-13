import torch
import math




# 注意力计算函数
def attention(query, key, value, dropout = None):
    '''
    '''
    
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = scores.softmax(dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn
