import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_data = datasets.MNIST(root="../dataset/mnist",train=True,transform=transforms.ToTensor,download=True)#transform是将图片的的读取状态转变为张量，本身是通过pil读取的
test_data = datasets.MNIST(root="../dataset/mnisr",train=False,transform=transforms.ToTensor,download=True)

train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True)