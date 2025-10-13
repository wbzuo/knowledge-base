import torch
import numpy as np
from torchvision import transforms

# 自定义数组图像
data = np.array([
                [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
                [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],
                [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]
        ],dtype='uint8')

print(f"without ToTensor shape:{data.shape}")
print(data)
data = transforms.ToTensor()(data)
print(f"ToTensor shape:{data.shape}")
print(data)

# ToTensor 作用   1.输入 H，W, C 转为 C H W 2. 像素点数据 除以255
