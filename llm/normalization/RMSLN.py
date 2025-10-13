import torch
import torch.nn as nn

class RMsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__() 
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数 g

    def _norm_(self, x): # 方法名建议去掉下划线，改为 _norm
        # 核心计算：x / RMS(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x): # 这里中文逗号，应改为英文逗号
        output = self._norm(x.float()).type_as(x) # 这里调用名应为 _norm_ 或修正为 _norm
        return output * self.weight