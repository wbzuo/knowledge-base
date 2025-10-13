import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        
        参数:
            data_dir (str): 数据存放目录路径
            transform (callable, optional): 可选的数据预处理/增强函数
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 获取所有图像文件的路径
        self.image_paths = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(data_dir, file_name))
                
        # 假设我们有一个简单的标签系统：文件名前缀为类别
        # 例如: "cat_001.jpg", "dog_123.jpg"
        self.labels = []
        for path in self.image_paths:
            file_name = os.path.basename(path)
            if file_name.startswith('cat'):
                self.labels.append(0)  # 猫为类别0
            elif file_name.startswith('dog'):
                self.labels.append(1)  # 狗为类别1
            else:
                self.labels.append(2)  # 其他为类别2
    
    def __len__(self):
        """
        返回数据集中的样本数量
        
        返回:
            int: 数据集中的样本总数
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        根据索引获取数据样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (图像张量, 标签)
        """
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
        
        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        
        return image, label