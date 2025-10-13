import torch
from torchvision import datasets , transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = datasets.MNIST(root='D:\datasets\mnist', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='D:\datasets\mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备：{ device }")
