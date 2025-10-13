# 构建模型
class Skip_gram(nn.Module):
    def __init__(self):
        super(Skip_gram, self).__init__()
        # W：one-hot到词向量的hidden layer
        self.W = nn.Parameter(torch.randn(voc_size, embedding_size).type((dtype)))
        # V：输出层的参数
        self.V = nn.Parameter(torch.randn(embedding_size, voc_size).type((dtype)))
 
    def forward(self, X):
        # X : [batch_size, voc_size] one-hot
        # torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V)  # output_layer : [batch_size, voc_size]
        return output_layer
model = Skip_gram().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 多分类，交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam优化算法