# 深度学习模型与数据处理笔记


## 1. 数据集处理
### 1.1 数据集划分
#### 1.1.1 划分方式
**内容**：使用 `train_test_split` 按比例划分训练集和验证集，确保数据分布均匀。
```python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
```

#### 1.1.2 路径处理
**内容**：数据集路径需明确，训练集和验证集路径分别设置，便于后续操作。
```python
train_dir = r'/image2/train'
val_dir = r'/image2/val'
```

### 1.2 数据集加载
#### 1.2.1 自定义数据集
**内容**：`ImageTxtDataset` 类通过 `txt` 文件加载图片路径和标签，灵活处理数据。
```python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.split()
                label = int(label.strip())
                self.labels.append(label)
                self.imgs_path.append(img_path)

```

#### 1.2.2 数据预处理
**内容**：包括调整图片大小、归一化等操作，确保输入模型的数据格式统一。
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```


## 2. 模型构建
### 2.1 AlexNet
#### 2.1.1 网络结构
**内容**：经典卷积神经网络架构，通过多层卷积、池化和全连接层实现图像分类。
```python
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
```

#### 2.1.2 输入输出
**内容**：输入为处理后的图像数据，输出为分类结果（`10` 类）。


### 2.2 Vision Transformer (ViT)
#### 2.2.1 网络结构
**内容**：基于 Transformer 架构，包括 Patch Embedding、Positional Embedding、Transformer Encoder 和 MLP Head。
```python
class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
```

你可以根据实际需求对上述文件进一步完善和调整，比如补充更多关于模型训练、测试相关的内容，或者添加一些使用示例等。如果还有其他修改建议或补充信息，欢迎随时告诉我。 