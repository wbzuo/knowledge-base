import torch.nn as nn
import numpy as np
import torch
import math


# Multi query attention
class MQA(nn.Module):
    def __init__(self, num_head, dimension_k, dimension_v, d_k, d_v, d_o):
        super().__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o
        self.fc_q = nn.Linear(dimension_k, num_head * d_k)
        self.fc_k = nn.Linear(dimension_k, d_k)
        self.fc_v = nn.Linear(dimension_v, d_v)
        self.fc_o = nn.Linear(num_head * d_v, d_o)
        self.softmax = nn.Softmax(dim=2)
        
        
    def forward(self, q, k, v, mask):
        
        batch, n_q, dimension_q = q.size()
        batch, n_k, dimension_k = k.size()
        batch, n_v, dimension_v = v.size()
          
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v) 
        
        q = q.view(batch, n_q, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)       
        k = k.repeat(self.num_head, 1, 1)
        v = v.repeat(self.num_head, 1, 1)
        
        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        mask = mask.repeat(self.num_head, 1, 1)
        attention = attention + mask
        attention = self.softmax(attention)
        
        output = torch.matmul(attention, v)
        output = output.view(self.num_head, batch, n_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)
        return attention, output
    
# 多头注意力  
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_head, dimension_k, dimension_v, d_k, d_v, d_o):
        super().__init__()
        self.num_head = num_head
        
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o
        
        self.fc_q = nn.Linear(dimension_k, num_head * d_k)
        self.fc_k = nn.Linear(dimension_k, num_head * d_k)
        self.fc_v = nn.Linear(dimension_v, num_head * d_v)
        
        self.fc_o = nn.Linear(num_head * d_v, d_o)
        
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask):
        batch, n_q, dimension_q = q.size()
        batch, n_k, dimension_k = k.size()
        batch, n_v, dimension_v = v.size()
        
        # 将q, k, v 投影进多头空间中
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        
        
        # 批处理进行矩阵乘法
        q = q.view(batch, n_q, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = k.view(batch, n_k, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = v.view(batch, n_v, self.num_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)
        
        # 缩放 点积注意力分数
        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        # 调整掩码的形状
        mask = mask.repeat(self.num_head, 1, 1)
        # 应用掩码 将不希望注意到的地方加上极小值
        attention = attention + mask
        # Softmax得到注意力权重
        attention = self.softmax(attention)
        
        # (nhead * batch nq, nk) * (nhead * batch , n_v, self.d_v)
        output = torch.matmul(attention, v)
        output = output.view(self.num_head, batch, n_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)
        
        
        return attention, output
        
        
        
        
batch = 10
n_q, n_k, n_v = 2, 4, 4 # sequence 长度
dimension_q, dimension_k, dimension_v = 128, 128, 64 # embedding的长度

num_head = 8


d_k, d_v, d_o = 16, 16, 8


q = torch.randn(batch, n_q, dimension_q)
k = torch.randn(batch, n_k, dimension_k)
v = torch.randn(batch, n_v, dimension_v)


mask  = torch.full((batch, n_q, n_k), -np.inf)    
mask = torch.triu(mask)


mha = MultiHeadAttention(num_head, dimension_k, dimension_v, d_k, d_v, d_o)
attention, output = mha(q, k, v, mask)
print(attention.size(), output.size())

mqa = MQA(num_head, dimension_k, dimension_v, d_k, d_v, d_o)
attention, output = mqa(q, k, v, mask)
print(attention.size(), output.size())
