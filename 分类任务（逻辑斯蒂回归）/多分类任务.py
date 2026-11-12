import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import torch.optim as optim

batch_size = 64

transformer = transforms.Compose(
    [
        transforms.ToTensor(),#转换为（c,w,h）
        transforms.Normalize((0.1307,),(0.3081,))#归一化
    ]
)#数据处理流水线，预定义对图像的操作

train_datas = datasets.MNIST(root="../dataset/mnist",train=True,download=True,transform=transformer)
train_loader = DataLoader(train_datas,batch_size=batch_size,shuffle=True,num_workers=4)
test_datas = datasets.MNIST(root="../dataset/mnist",train=True,download=False,transform=transformer)
test_loader = DataLoader(test_datas,batch_size=batch_size,shuffle=True,num_workers=4)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)
    def forward(self,x):
        x = x.view(-1,784)#将数据集压缩为两层张量    
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)#最后不做激活

model = Net()
#损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5) 

#构建训练函数
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('epoch: %d, batch: %5d, loss: %3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
#构建测试函数
def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            image,target = data
            outputs = model(image)
            _,predicted = torch.max(outputs.data,dim=1)#dim是沿着一维也就是第二维找找最大值
            total+=target.size(0)
            correct += (predicted==target).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))
if __name__ == '__main__':
    for epoch in range(100):
        train(epoch=epoch)
        if epoch%10 == 9:
            test()
