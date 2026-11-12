import torch

#1使用minibatch的方法构建数据集，即将所有x与y都构建为一个矩阵，可以实现批量运算
x_data = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0],[8.0]])
test = torch.tensor([5.0])

#2构建模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        #初始化父类
        super(LinearModel, self).__init__()
        #构造一个对象
        self.linear = torch.nn.Linear(1, 1)
        #这是一个线性层，两个参数分别是输入的维度和输出的维度

    def forward(self, x):
        #直接计算线性层
        y_pred = self.linear(x)
        return y_pred
#3构建损失函数(由于需要构建计算图，所以也是继承自model的)
model = LinearModel()

#class torch.nn.MSELoss(size_average=True, reduce=True)第一个参数，是否取batch均值，第二个参数是是否降维
criterion = torch.nn.MSELoss(size_average=False)
# torch.optim.SGD(params,lr=<object object>,momentum=0,dampening=0,weight_decay=0,nesterov=False )lr是学习率
#构建优化器，model.parameters()是modul这个父类中的方法，目的是检查所有的成员中的权重
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#4 训练循环
for epoch in range(1000):
    #计算y_hat
    y_pred = model(x_data)
    #计算损失函数
    loss = criterion(y_pred, y_data)#这里是一个对象
    #输出每一轮的损失
    print(epoch, loss)#当一个对象在打印的时候会自动调用__string__()函数

    #每次梯度归零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #进行更新
    optimizer.step()
#python的小技巧
#1
    #当构造函数的时候可以将函数中的参数变为  def func(*args,**kwargs)
    #这样当调用函数时可以向其中调用许多参数，只有1，2，“hello”这种原始数据的参数归纳为列表，而将x=1,y=2,这种的归纳为字典
#2
    #在python中构造类存在__call__()魔法方法，这种方法可以直接通过对象的名称调用，就比如self.linear(x),而在pytorch.nn的模型中这种方法会调用模型中的forward方法
    #这种类对象成为可调用的
# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[5.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)