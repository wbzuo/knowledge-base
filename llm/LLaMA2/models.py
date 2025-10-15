import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.eps = eps
    
    def _norm(self, x):
        # 这里sqrt()是平方根 rsqrt()为平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x):
        
        output = self._norm(x.float()).type_as(x)
        return self.weight * output

# RMSNorm测试代码
# torch.random.manual_seed(3427)

# input = torch.randn(1, 2, 4)
# norm = RMSNorm(4, 1e-5)



# print(input)
# print(f"shape of input:{ input.shape }")
# output = norm(input)
# print(output)
# print(f"shape of output:{output.shape}")