import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))   # 缩放参数
        self.beta = nn.Parameter(torch.zeros(features))   # 平移参数
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)   # 沿最后一个维度计算均值
        std = x.std(-1, keepdim=True)     # 沿最后一个维度计算标准差
        
        # 归一化： (x - mean) / (std + eps)
        normalized = (x - mean) / (std + self.eps)
        
        # 缩放和平移：gamma * normalized + beta
        return self.gamma * normalized + self.beta

# 简化测试
def quick_test():
    print("快速测试修复后的LayerNorm:")
    
    # 创建测试数据
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]]])
    
    custom_ln = LayerNorm(4)
    output = custom_ln(x)
    
    print(f"输入: {x}")
    print(f"输出: {output}")
    print(f"gamma: {custom_ln.gamma}")
    print(f"beta: {custom_ln.beta}")

quick_test()