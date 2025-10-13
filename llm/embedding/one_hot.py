import torch
import torch.nn.functional as F

labels = torch.tensor([0, 2, 1, 0])
print("原始标签", labels)

one_hot = F.one_hot(labels, num_classes = 3)
print("One-hot 编码结果:")
print(one_hot)
print("张量形状:", one_hot.shape)

def one_hot_scatter(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


labels = torch.tensor([0, 2, 1, 0])
num_classes = 3
result = one_hot_scatter(labels, num_classes)
print("使用 scatter_ 实现:")
print(result)