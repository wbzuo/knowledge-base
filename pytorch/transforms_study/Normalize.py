import torch
import numpy as np
from torchvision import transforms


# data.shape: [5, 5, 3]
data = np.array([
                [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
                [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],
                [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]
        ],dtype='uint8')

# 数据转化CHW 并归一化[0, 1]
data = transforms.ToTensor()(data)

# 模仿 batch 维度
data = torch.unsqueeze(data, 0)


nb_samples = 0
channel_mean = torch.zeros(3)
channel_std = torch.zeros(3)

N, C, H, W = data.shape[: 4]
data = data.view(N, C, -1)

channel_mean = data.mean(2).sum(0)
channel_std = data.std(2).sum(0)

nb_samples +=  N


channel_mean /= nb_samples
channel_std /= nb_samples


data = np.array([
                [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
                [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],
                [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]
        ],dtype='uint8')

data = transforms.ToTensor()(data)

data = transforms.Normalize(channel_mean, channel_std)(data)

print(data)

