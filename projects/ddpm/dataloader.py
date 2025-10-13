
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def load_transformed_dataset(img_size = 32, batch_size = 129) -> DataLoader:
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # load dataset
    train_dataset = torchvision.datasets.CIFAR10(root="./datasets",
                                                 train=True,
                                                 download=False,
                                                 transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets",
                                                 train=False,
                                                 download=False,
                                                 transform=test_transform)

    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # 将数据从[-1, 1]缩放到[0, 1]范围
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # 将通道顺序从CHW改为HWC
        transforms.Lambda(lambda t: t * 255.),  # 将数据缩放到[0, 255]范围
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # 将数据转换为uint8类型
        transforms.ToPILImage(),  # 将数据转换为PIL图像格式
    ])
    
    # 如果图像是批次数据,则取第一个图像
    if len(image.shape) == 4:
        image = image[0, :, :, :]
        
    return reverse_transforms(image)


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()
    image, _ = next(iter(train_loader))
    plt.imshow(show_tensor_image(image))
    plt.show()