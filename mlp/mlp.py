import torch
from torch import nn
import os

import torchvision
from torchvision.transforms import ToTensor

train_ds = torchvision.datasets.MNIST("/proj/intelisys-PG0/mnist/", train=True, transform=ToTensor(), download=True)
test_ds = torchvision.datasets.MNIST("/proj/intelisys-PG0/mnist/", train=False, transform=ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层输入展平后的特征长度28乘28,创建120个神经元
        self.liner_1 = nn.Linear(28*28, 120)
        # 第二层输入的是前一层的输出,创建84个神经元
        self.liner_2 = nn.Linear(120, 84)
        # 输出层接受第二层的输入84,输出分类个数10
        self.liner_3 = nn.Linear(84, 10)

    def forward(self, input):
        x = input.view(-1, 28*28)  # 将输入展平为二维(1,28,28)->(28*28)
        x = torch.relu(self.liner_1(x))
        x = torch.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x
    
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
'''
注意两个参数
1. weight: 各类别的权重（处理不平衡数据集）
2. ignore_index: 忽略特定类别的索引
另外,它要求实际类别为数值编码,而不是独热编码
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
# print(device)  # 可选：打印使用的设备

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取当前数据集样本总数量
    num_batches = len(dataloader)   # 获取当前data loader总批次数
    
    # train_loss用于累计所有批次的损失之和, correct用于累计预测正确的样本总数
    train_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # 进行预测,并计算当前批次的损失
        pred = model(X)
        loss = loss_fn(pred, y)
        # 利用反向传播算法,根据损失优化模型参数
        optimizer.zero_grad()   # 先将梯度清零
        loss.backward()          # 损失反向传播,计算模型参数梯度
        optimizer.step()         # 根据梯度优化参数
        
        with torch.no_grad():
            # correct用于累计预测正确的样本总数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # train_loss用于累计所有批次的损失之和
            train_loss += loss.item()
            
    # train_loss 是所有批次的损失之和,所以计算全部样本的平均损失时需要除以总的批次数
    train_loss /= num_batches
    # correct 是预测正确的样本总数,若计算整个epoch总体正确率,需要除以样本总数量
    correct /= size
    return train_loss, correct

def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct

# 对全部的数据集训练50个epoch(一个epoch表示对全部数据训练一遍)
epochs = 50 
train_loss, train_acc = [], []
test_loss, test_acc = [], []

for epoch in range(epochs):
    # 调用train()函数训练
    epoch_loss, epoch_acc = train(train_dl, model, loss_fn, optimizer)
    # 调用test()函数测试
    epoch_test_loss, epoch_test_acc = test(test_dl, model)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    
    # 定义一个打印模板
    template = ("epoch:{:2d},train_loss:{:.6f},train_acc:{:.1f}%,""test_loss:{:.5f},test_acc:{:.1f}%")
    print(template.format(epoch, epoch_loss, epoch_acc*100, epoch_test_loss, epoch_test_acc*100))

print("Done")