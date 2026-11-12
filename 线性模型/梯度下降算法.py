import numpy as np
import matplotlib.pyplot as plt

#设置参数
#学习率
a = 0.01
#权重
w = 1
#原始数据
x_list = [1,2,3,4]
y_list = [2,4,6,8]
epochs = []
costs = []

#定义线性模型
def forward(x):
    return w*x
#定义损失函数模型
def cost(xs,ys):
    costs = 0
    for x,y in zip(xs,ys):
        loss = (forward(x)-y)**2
        costs += loss
    costs /= len(xs)
    return costs
#定义梯度函数
def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w-y)
    grad /= len(xs)
    return grad

print("predict:",5,forward(5))
#开始训练
for epoch in range(100):
    cost_val = cost(x_list, y_list)
    grad_val = gradient(x_list, y_list)
    w -= a * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epochs.append(epoch)
    costs.append(cost_val)
print('Predict (after training)', 5, forward(5))
plt.plot(epochs,costs)
plt.ylabel("cost")
plt.xlabel("w")
plt.show()

