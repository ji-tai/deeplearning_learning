import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from pathlib import Path


class InceptionA(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.branch1_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

		self.branch2_1 = nn.Conv2d(in_channels, 16, kernel_size=1)

		self.branch3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
		self.branch3_5 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

		self.branch4_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
		self.branch4_3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
		self.branch4_3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

	def forward(self, x):
		branch1 = self.branch1_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))

		branch2 = self.branch2_1(x)

		branch3 = self.branch3_5(self.branch3_1(x))

		branch4 = self.branch4_3_2(self.branch4_3_1(self.branch4_1(x)))

		return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.incep1 = InceptionA(in_channels=10)
		self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
		self.incep2 = InceptionA(in_channels=20)
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.fc = nn.Linear(88 * 4 * 4, 10)

	def forward(self, x):
		batch_size = x.size(0)
		x = F.relu(self.pool(self.conv1(x)))
		x = self.incep1(x)
		x = F.relu(self.pool(self.conv2(x)))
		x = self.incep2(x)
		x = x.view(batch_size, -1)
		return self.fc(x)


def train(model, data_loader, criterion, optimizer, device, epoch):
#   将模型设置为训练模式
    model.train()
    total_loss = 0.0

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
		#可以输出outputs的形状
        print(f"Outputs shape: {outputs.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_index % 300 == 0:
            print('epoch: %d, batch: %5d, loss: %.4f' %
				  (epoch, batch_index, total_loss / batch_index))

    average_loss = total_loss / len(data_loader)
    print('epoch: %d, average loss: %.4f' % (epoch, average_loss))
    return average_loss


def test(model, data_loader, device):
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in data_loader:
			images, labels = images.to(device), labels.to(device)
			predictions = model(images).argmax(dim=1)
			total += labels.size(0)
			correct += (predictions == labels).sum().item()

	accuracy = 100 * correct / total
	print('Accuracy on test set: %.2f %%' % accuracy)
	return accuracy


def plot_training_history(loss_history, accuracy_history):
	epochs = range(1, len(loss_history) + 1)
	figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

	axes[0].plot(epochs, loss_history, marker='o', linewidth=2, color='tab:blue')
	axes[0].set_ylabel('Loss')
	axes[0].set_title('Training Loss')
	axes[0].grid(alpha=0.3)

	axes[1].plot(epochs, accuracy_history, marker='o', linewidth=2, color='tab:green')
	axes[1].set_xlabel('Epoch')
	axes[1].set_ylabel('Accuracy (%)')
	axes[1].set_title('Test Accuracy')
	axes[1].grid(alpha=0.3)

	plt.xticks(list(epochs))
	figure.tight_layout()
	plt.show()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dataset_root = Path(__file__).resolve().parents[1] / 'dataset' / 'mnist'
	transformer = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	train_data = MNIST(root=dataset_root, train=True,
					   transform=transformer, download=True)
	test_data = MNIST(root=dataset_root, train=False,
					  transform=transformer, download=True)
	train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
	test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

	model = Net().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

	loss_history = []
	accuracy_history = []
	for epoch in range(1, 11):
		loss_history.append(train(model, train_loader, criterion, optimizer, device, epoch))
		accuracy_history.append(test(model, test_loader, device))

	plot_training_history(loss_history, accuracy_history)
