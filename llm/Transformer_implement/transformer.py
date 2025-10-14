import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import math


@dataclass
class ModelArgs:
    n_embd: int
    n_heads: int
    dim: int
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int


args = ModelArgs(
        n_embd=512,
        n_heads=8,
        dim=512,
        dropout=0.1,
        max_seq_len=1024,
        vocab_size=30522,  # BERT词汇表大小
        block_size=512,
        n_layer=12
    )


# FNN
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x):
        x = F.relu(self.w1(x))
        x = self.w2(x)
        x = self.dropout(x)
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal = False):
        super().__init__()
        
        # assert: 判断向量维数 是 头的整数倍数
        assert args.dim % args.n_heads == 0
        # 模型并行处理大小，默认为1
        model_parallel_size = 1
        # 本地计算头数，等于总头数
        self.n_local_heads = args.n_heads
        # 每个头的纬度
        self.head_dim = args.dim // args.n_heads
        
        
        # Wq, Wk, Wv参数矩阵，每个参数矩阵为 n_embd * n_embd
        self.wq = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias = False)
        
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)
        
        # 注意力的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        
        # 残差连接的dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal
        
        # 创建一个上三角矩阵，用于遮蔽未来信息
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal = 1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)  
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, seqlen, _ = q.shape
        
        # 计算查询Q， K，V
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        
        
        # 拆成多头注意力
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        # [B, nh, T, hs]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 掩码注意力机制
        if self.is_causal:
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
                            
        # 计算 softmax [bsz, nh, T, T]
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        # 计算dropout  
        scores = self.attn_dropout(scores)
        # [bsz, nh, T, T] * [bsz, nh, T, hs]
        output = torch.matmul(scores, xv)
        
        # 恢复时间纬度并合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # [bsz, T, C] * [bsz, T, dim]
        output = self.wo(output)
        output = self.resid_dropout(output)
        
        return output
        
        
# # MultiHeadAttention test code
# args = ModelArgs(
#         n_embd=512,
#         n_heads=8,
#         dim=512,
#         dropout=0.1,
#         max_seq_len=1024,
#         vocab_size=10000,
#         block_size=1024,
#         n_layer=6
#     )


# attention = MultiHeadAttention(args, is_causal=True)
# batch_size, seq_len = 2, 10
# input = torch.randn(batch_size, seq_len, args.dim)
# print(f"输入形状: {input.shape}")
    
# # 前向传播
# output = attention(input, input, input)
# print(f"输出形状: {output.shape}")



class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True) # 修正：keepdim 而不是 keep_dim
        std = x.std(-1, keepdim = True)
        
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
    

# batch_size, seq_len, features = 4, 8, 512
# input = torch.randn(batch_size, seq_len, features)
# # print(f"输入形状: {input.shape}")
    
# # # 前向传播
# layer_norm = LayerNorm(features)
# output = layer_norm(input)
# print(f"输出形状: {output.shape}")


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 第一个子层：自注意力 + LayerNorm (Pre-Norm结构)
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal= False)
        
        
        # 第二个子层：前馈神经网络 + LayerNorm (Pre-Norm结构)  
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)
        
    def forward(self, x):
        
        # Pre-Norm: 先LayerNorm，再自注意力，最后残差连接
        x = self.attention_norm(x)
        h  = x +  self.attention.forward(x, x, x)

        
        
        
        # FNN
        h_norm = self.fnn_norm(h)
        fnn_output = self.feed_forward.forward(h_norm)
        out = h + fnn_output
        
        return out
        

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)
    
    def forward(self, x):
        # 通过N层
        for layer in self.layers:
            x = layer(x)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 第一个子层：掩码注意力 + LayerNorm (Pre-Norm结构)
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args, is_causal= True)
        
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal= False)
        
        # 第二个子层：前馈神经网络 + LayerNorm (Pre-Norm结构)  
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)
        
    def forward(self, x, enc_out):
        
        x = self.attention_norm_1(x)
        # 掩码注意力机制
        x = x + self.mask_attention.forward(x, x, x)
        
        # 多头注意力机制
        x = self.attention_norm_2(x)
        h = x + self.attention.forward(x, enc_out, enc_out)
        
        # 经过FNN
        out = h + self.feed_forward.forward(self.fnn_norm(x))
        return out

class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


# # 创建编码器层
# # encoder_layer = EncoderLayer(args)
# # encoder = Encoder(args)
# decoder_layer = DecoderLayer(args)
# # decoder = Decoder(args)
# batch_size, seq_len, features = 4, 8, 512
# input = torch.randn(batch_size, seq_len, features)
# enc_out = torch.randn(batch_size, seq_len, features)
# # print(f"输入形状: {input.shape}")
    
# # # # 前向传播
# # layer_norm = LayerNorm(features)
# # output = encoder_layer(input)
# # output = encoder(input)
# output = decoder_layer(input, enc_out)
# # output = decoder(input, enc_out)
# print(f"输出形状: {output.shape}")




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


# @dataclass
# class ModelArgs:
#     n_embd: int
#     n_heads: int
#     dim: int
#     dropout: float
#     max_seq_len: int
#     vocab_size: int
#     block_size: int
#     n_layer: int

class Transfomer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args)
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)
        
        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
    
    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
        '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss