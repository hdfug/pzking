import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# 1. 数据准备
def prepare_data(data_dir='C:/Users/彭子昂/PycharmProjects/suanfa3/daythree/Images', batch_size=64, val_ratio=0.2):
    # 数据增强和归一化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载完整数据集
    full_dataset = ImageFolder(root=data_dir, transform=transform)

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, full_dataset.classes


# 2. 定义神经网络模型 (使用ResNet50预训练模型)
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(ImageClassifier, self).__init__()
        # 使用预训练的ResNet50
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # 冻结所有卷积层 (可选)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # 替换最后的全连接层
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# 3. 训练函数
def train_model(train_loader, val_loader, num_classes, epochs=25, lr=0.001):
    # 初始化模型并移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # TensorBoard记录
    writer = SummaryWriter()

    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)

        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': epoch_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': epoch_acc, 'val': val_acc}, epoch)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    writer.close()
    return model


# 4. 评估函数
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(data_loader, desc='Validating'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(data_loader.dataset)
    total_acc = running_corrects.double() / len(data_loader.dataset)

    return total_loss, total_acc


# 5. 主函数
def main():
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"🚀 GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 将使用CPU训练，速度会较慢")

    # 准备数据
    train_loader, val_loader, classes = prepare_data()
    print(f"📊 数据集包含 {len(classes)} 个类别")

    # 训练模型
    trained_model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(classes),
        epochs=25,
        lr=0.001
    )

    print("✅ 训练完成!")


if __name__ == '__main__':
    main()#