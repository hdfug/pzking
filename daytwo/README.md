# 项目代码总结

## 一、图像分类模型测试代码
```python
import torch
import torchvision
from PIL import Image
from model import *
from torchvision import transforms

image_path = "./Image/img.png"
image = Image.open(image_path)
print(image)
# png格式是四个通道，除了RGB三个通道外，还有一个透明通道,调用image = image.convert('RGB')
image = image.convert('RGB')

# 改变成tensor格式
trans_reszie = transforms.Resize((32, 32))
trans_totensor = transforms.ToTensor()
transform = transforms.Compose([trans_reszie, trans_totensor])
image = transform(image)
print(image.shape)

# 加载训练模型（使用CPU）
device = torch.device("cpu")
model = torch.load("model_save\\chen_9.pth", map_location=device)

# print(model)

image = torch.reshape(image, (1, 3, 32, 32))

# 将模型转换为测试模型
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))
```
## 二、CIFAR10 数据集训练代码
```python
# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 创建网络模型

chen = Chen()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(chen,f"model_save\\chen_{i}.pth")
    print("模型已保存")

writer.close()
```
## AlexNet 模型训练代码
```python
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(48, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 2048),
            torch.nn.Linear(2048, 2048),
            torch.nn.Linear(2048, 10)
        )

    def forward(self, x):
        return self.model(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = AlexNet()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

total_train_step = 0
total_test_step = 0
epoch = 10
writer = SummaryWriter("../logs_train")

start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    model.train()
    for data in train_loader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / len(test_data)}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

    torch.save(model, f"model_save\\alexnet_{i}.pth")
    print("模型已保存")

writer.close()
```