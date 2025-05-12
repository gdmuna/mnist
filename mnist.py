import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import os

# 定义超参数
epochs = 10
batch_size = 64
lr = 0.001

# 设置数据转换方式
transform = transforms.Compose([
    transforms.ToTensor(),  # 把数据转换为张量（Tensor）
    transforms.Normalize(  # 标准化，即使数据服从期望值为 0，标准差为 1 的正态分布
        mean=[0.5, ],  # 期望
        std=[0.5, ]  # 标准差
    )
])

# 训练集导入
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
# 测试集导入
data_test = datasets.MNIST(root='data/', transform=transform, train=False)

# 数据装载
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

# 构建卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一层卷积层
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二卷积层
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x

# 定义求导函数
def get_Variable(x):
    x = torch.autograd.Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# 定义网络
cnn = CNN()

# 判断是否有可用的 GPU 以加速训练
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 设置损失函数为 CrossEntropyLoss（交叉熵损失函数）
loss_F = nn.CrossEntropyLoss()

# 设置优化器为 Adam 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

# 训练
for epoch in range(epochs):
    running_loss = 0.0  # 一个 epoch 的损失
    running_correct = 0.0  # 准确率
    print("Epoch [{}/{}]".format(epoch, epochs))
    for data in train_loader:
        # DataLoader 返回值是一个 batch 内的图像和对应的 label
        X_train, y_train = data
        X_train, y_train = get_Variable(X_train), get_Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)
        
        optimizer.zero_grad()  # 梯度置零
        loss = loss_F(outputs, y_train)  # 求损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新所有梯度
        
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0.0

    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = get_Variable(X_test), get_Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
        running_loss / len(data_train), 100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)))

# 保存模型
os.makedirs('model_save', exist_ok=True)  # 自动创建目录（如果不存在）
torch.save(cnn, os.path.join('model_save', 'model.pth'))  # 修改保存路径