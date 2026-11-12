#获取官方已有的数据集
import torch
import numpy as np
import torch.nn.functional as F

#数据集
xy = np.loadtxt("diabetes.csv.gz",delimiter=",",dtype=np.float32)
x_datas = torch.from_numpy(xy[: , :-1])
y_datas = torch.from_numpy(xy[: , [-1]])
#对于二维数组，第一个操作数是行操作，第二个操作数是列操作，同理在多维数组中，有n维，就会有n个操作数
#直接用：说明取所有行，-1指最后一行，而[-1]是在保留原来的维度，否则只取一层会降低一个维度
# --------------------------------------------------

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        y_hat = self.activate(self.linear3(x))
        return y_hat
        


model = Model()

# --------------------------------------------------

criterion = torch.nn.BCELoss(size_average=True)
#这个size_average就是将一个batch中的loss求取平均值
optimizer = torch.optim.Rprop(model.parameters(), lr=0.1)

# --------------------------------------------------

for epoch in range(10000):

#前馈
    y_pred = model(x_datas)
    loss = criterion(y_pred, y_datas)

    print(epoch, loss.item())
#反馈
    optimizer.zero_grad()
    loss.backward()
#更新
    optimizer.step()
import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 200)

# x_t = torch.Tensor(x).view((200, 1))

# y_t = model(x_t)

# y = y_t.data.numpy()

# plt.plot(x, y)
# plt.plot([0, 10], [0.5, 0.5], c='r')
# plt.xlabel('Hours')
# plt.ylabel('Probability of Pass')
# plt.grid()
# plt.show()
