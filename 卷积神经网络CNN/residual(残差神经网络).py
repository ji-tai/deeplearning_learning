import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Resblock(nn.Module):
	def __init__(self, input_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)

	def forward(self, x):
		residual = x
		x = F.relu(self.conv1(x))
		x = self.conv2(x)
		x += residual
		return F.relu(x)


class ResNet(nn.Module):
	def __init__(self, input_channels=1, num_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.resblock1 = Resblock(64)
		self.resblock2 = Resblock(64)
		self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(64, num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.pool1(x)
		x = self.resblock1(x)
		x = self.resblock2(x)
		x = self.pool2(x)
		x = torch.flatten(x, 1)
		return self.fc(x)


def train(model, data_loader, criterion, optimizer, device, epoch):
	model.train()
	total_loss = 0.0

	for batch_index, (images, labels) in enumerate(data_loader, start=1):
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
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

	axes[1].set_xticks(list(epochs))
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

	model = ResNet().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

	loss_history = []
	accuracy_history = []
	for epoch in range(1, 11):
		loss_history.append(train(model, train_loader, criterion, optimizer, device, epoch))
		accuracy_history.append(test(model, test_loader, device))

	plot_training_history(loss_history, accuracy_history)
