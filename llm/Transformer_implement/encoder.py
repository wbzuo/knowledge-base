import torch
import torch.nn as nn
import torch.nn.functional as F

# 缺少这些类的定义
class MultiHeadAttention(nn.Module):
    pass

class MLP(nn.Module):
    pass

class LayerNorm(nn.Module):  # 虽然您前面定义了，但需要确保正确
    pass

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)
    
    def forward(self, x):
        norm_x = self.attention_norm(x)
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        out = h + self.feed_forward.forward(self.fnn_norm(h))

        return out
    
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)
        
    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
        