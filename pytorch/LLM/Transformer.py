
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random


# word_embdding 以序列建模为例
# 考虑source sentence 和 target  sentence

# 构建序列 以索引形式表示
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8


max_src_seq_len = 4
max_tgt_seq_len = 4

max_position_len = 5

batch_size = 2
src_len = torch.randint(2, 5, (batch_size, ))
tgt_len = torch.randint(2, 5, (batch_size, ))
print(src_len, tgt_len)


src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len - L)), 0) for L in  src_len]) # 单词索引构成的句子序列
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_tgt_seq_len - L)), 0) for L in  tgt_len])
# 训练数据
print(src_seq, tgt_seq)


# 构造embedding
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)

print(src_embedding_table.weight)



src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)

print(src_embedding)



# position embdding

post_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / model_dim)

pe_embbding_table = torch.zeros(max_position_len, model_dim)

pe_embbding_table[:, 0::2] = torch.sin(post_mat / i_mat)
pe_embbding_table[:, 1::2] = torch.cos(post_mat / i_mat)


print(post_mat)
print(i_mat)


print(pe_embbding_table)

pe_embbding = nn.Embedding(max_position_len, model_dim)
pe_embbding.weight = nn.Parameter(pe_embbding_table, requires_grad=False)


src_pos = torch.tensor(torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]))
tgt_pos = torch.tensor(torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]))
print(src_pos)
print(tgt_pos)

src_pe_embedding = pe_embbding(src_pos)
tgt_pe_embedding = pe_embbding(tgt_pos)
print(src_pe_embedding)
print(tgt_pe_embedding)




# encoder self-attention mask
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_src_seq_len - L)), 0) for L in src_len]),2)

valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1- valid_encoder_pos_matrix
print(valid_encoder_pos_matrix)
print(invalid_encoder_pos_matrix)


mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

print(mask_encoder_self_attention)

score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -np.inf)



