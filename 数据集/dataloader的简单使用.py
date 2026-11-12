from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. 准备数据集
class Mydataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(
            filepath,
            delimiter=",",
            dtype=np.float32
            # skiprows=1
        )
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 2. 构建模型
class Mymodel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 6)
        self.linear3 = torch.nn.Linear(6, 1)

        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        y_hat = self.activate(self.linear3(x))

        return y_hat


if __name__ == '__main__':

    # 1. 创建数据集
    dataset = Mydataset("diabetes.csv.gz")

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # 2. 创建模型
    model = Mymodel()

    # 3. 创建损失函数
    criterion = torch.nn.BCELoss()

    # 4. 创建优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    # 用来保存每一个 epoch 的平均 loss
    loss_list = []

    # 5. 训练循环
    for epoch in range(100):

        # 保存当前 epoch 中所有 batch 的 loss
        epoch_loss = 0

        for i, data in enumerate(train_loader, 0):

            x, y = data

            # forward
            y_hat = model(x)

            loss = criterion(y_hat, y)

            print(epoch, i, loss.item())

            # 把当前 batch 的 loss 累加起来
            epoch_loss += loss.item()

            # backward
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        # 当前 epoch 所有 batch 的平均 loss
        average_loss = epoch_loss / len(train_loader)

        # 保存当前 epoch 的平均 loss
        loss_list.append(average_loss)

        print("Epoch:", epoch, "Average Loss:", average_loss)

    # 6. 绘制 Loss 曲线
    plt.plot(range(1, 101), loss_list)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid()

    plt.show()