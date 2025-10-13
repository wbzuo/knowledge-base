import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim

# Define transformations (convert images to tensors and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load the datasets
train_dataset = MNIST(root=r"D:\datasets\mnist", train=True, download=True, transform=transform)
test_dataset = MNIST(root=r"D:\datasets\mnist", train=False, download=True, transform=transform)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create data loaders for batching
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Example: Access a sample
# sample_image, sample_label = train_dataset[0]
# print(f"Sample image shape: {sample_image.shape}")
# print(f"Sample label: {sample_label}")

# build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1048)
        self.fc2 = nn.Linear(1048, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = Net()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


model.train()
epochs = 10
for epoch in range(epochs):
# Example: Iterate through batches
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
    print(f'Epoch: {epoch}, Average Loss: {total_loss/len(train_loader):.4f}')
    
    
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')