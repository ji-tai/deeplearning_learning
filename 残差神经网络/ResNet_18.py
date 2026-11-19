"""
(B, C, H, W) = (Batch_Size, 3, 224, 224)训练数据
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 构建数据集

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# 定义残差块

class BasicBlock(nn.Module):
    expansion = 1# 卷积层最后输出的通道数与out_channels的商，注意是类属性
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , stride = stride , kernel_size = 3 , bias = False , padding = 1)
        # 这个模块无法复用
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels = out_channels , out_channels = out_channels , stride = 1 , kernel_size = 3 , bias = False , padding = 1)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride,bias = False),nn.BatchNorm2d(out_channels))
    def forward(self,x):
        # 主分支
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        out = self.relu(self.shortcut(x)+out)
        return out

# 定义残差网络

class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):# num_classes是分类类别
        super().__init__()
        self.in_channels = 64# 这是第一次卷积之后的数字
        self.conv1 = nn.Conv2d(3, 64,kernel_size=7,stride=2,padding=3,bias=False)# (Batch_size,64,224,224)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # 四个残差阶段
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        # 这里后三个残差阶段中只有第一个残差块中的步数变为的了2
        self.layer2 = self.make_layer(block, 128, 2, stride=2)

        self.layer3 = self.make_layer(block, 256, 2, stride=2)

        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def make_layer(self,block,out_channels,blocks,stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))
        return nn.Sequential(*layers)
    def forward(self,x):# (Batch_size,3,112,112)
        # 初始卷积
        x = self.conv1(x)# (Batch_size,64,224,224)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)# (Batch_size,64,112,112)

        # 四个残差阶段
        x = self.layer1(x)# (Batch_size,64,112,112)
        x = self.layer2(x)# (Batch_size,128,56,56)
        x = self.layer3(x)# (Batch_size,256,28,28)
        x = self.layer4(x)# (Batch_size,512,14,14)

        # 平均池化
        x = self.avgpool(x)# (Batch_size,512,1,1)

        # 展平
        x = torch.flatten(x, 1)# (Batch_size,512)

        # 分类
        x = self.fc(x)
        return x

# 创建模型

def resnet18(num_classes=10):

    return ResNet(
        BasicBlock,
        num_classes=num_classes
    )


# 创建模型
model = resnet18(num_classes=10)

# 训练模型
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

# 2. 损失函数和优化器
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 3. 训练
for epoch in range(10):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

