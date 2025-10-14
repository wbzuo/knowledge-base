import torch
import torch.nn
import torch.nn.functional as F


# x = x + MultiHeadSelfAttention(LayerNorm(x))
# output = x + FNN(LayerNorm(x))


# 注意力计算
# h = x + self.attention.forward(self.attention_norm(x))
# # 经过前馈神经网络
# out = h + self.feed_forward.forward(self.fnn_norm(h))


pass