import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()  
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层
        
        # 计算位置编码并将其存储在pe张量中
        pe = torch.zeros(max_len, d_model)                # 创建一个max_len x d_model的全零张量
        # 生成0到max_len-1的整数序列，并添加一个维度
        position = torch.arange(0, max_len).unsqueeze(1)  
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
 
        # 对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # 在第一个维度添加一个维度，以便进行批处理
        
    # 定义前向传播函数
    def forward(self, x):
        # 将输入x与对应的位置编码相加
        x = x + self.pe[:, : x.size(1)]
        # 应用dropout层并返回结果
        return self.dropout(x)