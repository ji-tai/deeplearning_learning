from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


class CNN(nn.Module):
	"""用于 MNIST 手写数字分类的卷积神经网络。"""
	"""输入数据集是28*28*1,torch.Size([1, 28, 28])"""

	def __init__(self):
		super().__init__()
		# nn.Sequuential是一个按照顺序运行的模块
		self.features = nn.Sequential(
			# 卷积神经网络
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),# 28*28*32
			# 归一化处理，将数据的分布拉回到了均值趋近于0，方差趋近于1的位置
			nn.BatchNorm2d(32),# 28*28*32
			# 非线性函数
			nn.ReLU(inplace=True),# 28*28*32
			# 最大池化层
			nn.MaxPool2d(kernel_size=2),# 14*14*32
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),# 14*14*64
			nn.BatchNorm2d(64),# 14*14*64
			nn.ReLU(inplace=True),# 14*14*64
			nn.MaxPool2d(kernel_size=2),# 7*7*64
		)
		# 特征提取层
		self.classifier = nn.Sequential(
			# flatten层将多维的数据转换为一维，将多维的数据转换为一层
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			# inplace这个参数是直接在原来的输入数据上做修改，不额外生成其他的数据
			nn.ReLU(inplace=True),
			# dropout是一个正则化层，训练时，对输入张量的每个元素，以概率 p 随机置 0，其余元素按 1/(1-p) 放大。防止过拟合
			nn.Dropout(p=0.3),
			nn.Linear(128, 10),
		)
		# 经典分类层
	def forward(self, images):
		features = self.features(images)
		return self.classifier(features)
# nn.Sequential是一个神经网络的容器，直接按照顺序进行网络的运行


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch):
	# 将模型开启训练模式
	model.train()
	total_loss = 0.0

	for batch_index, (images, labels) in enumerate(data_loader, start=1):
		# 将取出的数据全部都放到指定设备中
		images = images.to(device)
		labels = labels.to(device)

		# 每一此的训练过程
		# 更新器清零
		optimizer.zero_grad()
		# logits 是深度学习里非常常见的术语，指的是模型最后一层输出的、未经 softmax/sigmoid 等归一化的原始分数
		# 进行前向传播
		logits = model(images)
		loss = loss_function(logits, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		if batch_index % 200 == 0:
			print(
				f"Epoch {epoch}/{EPOCHS}, batch {batch_index}/{len(data_loader)}, "
				f"loss: {total_loss / batch_index:.4f}"
			)

	return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
	# 将模型开启测试模式
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in data_loader:
			images = images.to(device)
			labels = labels.to(device)
			predictions = model(images).argmax(dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.size(0)

	return correct / total


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# 构建数据集
	dataset_root = Path(__file__).resolve().parents[1] / "dataset" / "mnist"
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
		]
	)
	train_data = MNIST(
		root=dataset_root, train=True, transform=transform, download=True
	)
	test_data = MNIST(
		root=dataset_root, train=False, transform=transform, download=True
	)
	train_loader = DataLoader(
		train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
	)
	test_loader = DataLoader(
		test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
	)

	# 定义模型
	model = CNN().to(device)
	# 定义损失器
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	for epoch in range(1, EPOCHS + 1):
		average_loss = train_one_epoch(
			model, train_loader, loss_function, optimizer, device, epoch
		)
		accuracy = evaluate(model, test_loader, device)
		print(
			f"Epoch {epoch}/{EPOCHS}: loss={average_loss:.4f}, "
			f"test accuracy={accuracy * 100:.2f}%"
		)


if __name__ == "__main__":
	main()
