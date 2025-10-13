
# 目标数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签


import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x_data[idx], dtype = torch.float32)
        y = torch.tensor(self.y_data[idx], dtype = torch.float32)
        
        return x, y

dataset = MyDataset(x_data=X_data, y_data=Y_data)

data_loader = DataLoader(dataset, batch_size = 2, shuffle = True)

for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')

