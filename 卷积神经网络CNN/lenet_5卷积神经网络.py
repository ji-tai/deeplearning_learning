# 1.1卷积层
    #一般卷积核的张量大小为4维，m*n*w*h,---m:输出的数据的通道数；---n:输入的数据的通道数；w,h：卷积核的大小，
    #torch.tensor()将python的数据转换为tensor的形式
    #torch.randn()创建规定大小的tensor格式的数据，并用【0，1】之间的随机数据填充她
    # torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)：要转换的数据，指定数据类型，指定数据创建的位置（GPU/CPU），是否需要计算梯度，锁页内存
    # torch.rand(*size, dtype=None, device=None, requires_grad=False, pin_memory=False)
    # torch.tensor(data).view(size)改变tensor的数据结构 
    # 创建卷积神经网络层：conv_layer = torch.nn.Conv2d（n,m,kernel = 3,padding = 1,stride = 2,bias = False）输出，输入，卷积核的大小，是否在原来的数据上加上空白数据，卷积核移动的步长，偏置值
    # 自己初始化卷积层的数据：kernel = torch.tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3);conv_layer.weight.data = kernel.data
# 2.2下采样
    # pooling
        # >>maxpooling：maxpooling是在自己的卷积核上取最大值，一般来说步长等于卷积核的长宽
        # >>Average Pooling
from torchvision.datasets import MNIST 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose(
    [
        transforms.ToTensor(),#转换为（c,w,h）
        transforms.Normalize((0.1307,),(0.3081,))#归一化
    ]
)

train_data = MNIST(root="../dataset/mnist",train=True,transform=transformer,download=True)
test_data = MNIST(root="../dataset/mnist",train=False,transform=transformer,download=True)
train_dataload = DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
test_dataload = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0)

#构建模型
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5,stride=1)#10*24*24
        self.maxpooling1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)#10*12*12
        self.conv_layer2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5,stride=1)#20*8*8
        self.fc1 = torch.nn.Linear(320,160)
        self.fc2 = torch.nn.Linear(160,80)
        self.fc3 = torch.nn.Linear(80,10)
    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.maxpooling1(self.conv_layer1(x)))
        x = F.relu(self.maxpooling1(self.conv_layer2(x)))
        x = x.view(batch_size,-1)#-1是自动计算大小
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return x
#criterion = torch.nn.CrossEntropyLoss()使用这个损失函数时不需要最后一层做激活
model = CNN()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5) 
# 构建训练函数
def train(epoch):
    #统计loss值
    running_loss = 0.0
    batch_losses = []
    #进入batch循环

    for batch_idx,data in enumerate(train_dataload,0):
        #取数据
        image,label = data
        #将数据放到GPU上
        image = image.to(device)
        label = label.to(device)
        #梯度清零
        optimizer.zero_grad()
        #前向传播
        outputs = model(image)
        loss = criterion(outputs,label)
        #反向传播
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        batch_losses.append(loss_value)
        running_loss += loss_value
        if batch_idx % 300 == 299:
            print('epoch: %d, batch: %5d, loss: %3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    return batch_losses
def test():
    #总共的图片数
    total = 0
    #正确的图片数量
    correct = 0
    #不进行梯度的计算
    with torch.no_grad():

        for data in test_dataload:
            image,label = data
            #将数据放到GPU上
            image = image.to(device)
            label = label.to(device)
            #直接前向传播
            outputs = model(image)
            _,predicted = torch.max(outputs,dim=1)#第一维是batch,后面是实际存储的结果函数
            total += label.size(0)
            #如果tensor结构进行比较，（>,<,=）会生成一个新的tensor结构里面的值分别是Ture,Fase
            correct += (label==predicted).sum().item()#.item()就是从tensor中获得python数据
        print('Accuracy on test set: %d %%' % (100 * correct / total))


def plot_loss_history(loss_history):
    """绘制 batch loss 随训练轮次变化的三维曲线。"""
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection='3d')

    for epoch, batch_losses in enumerate(loss_history, start=1):
        batch_numbers = range(1, len(batch_losses) + 1)
        epoch_numbers = [epoch] * len(batch_losses)
        axis.plot(batch_numbers, epoch_numbers, batch_losses, linewidth=1)

    axis.set_xlabel('Batch')
    axis.set_ylabel('Epoch')
    axis.set_zlabel('Loss')
    axis.set_title('Training Loss by Batch and Epoch')
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    loss_history = []
    for epoch in range(10):
        loss_history.append(train(epoch=epoch))
        if epoch%10 == 9:
            test()
    plot_loss_history(loss_history)