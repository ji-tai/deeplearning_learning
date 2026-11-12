import numpy as np
import matplotlib.pyplot as plt

x_data = [1,2,3,4]
y_data = [2,4,6,8]

#定义模型函数
def forward(x):
    return x*w

#定义损失函数
def loss(x,y):
    y_hat = forward(x)
    return (y_hat-y)*(y_hat-y)

w_list = []
MSE_list = []

#通过穷举法不断尝试权重值
for w in np.arange(0.0,4.0,0.1):
    number = 0
    cost = 0
    print("W = ",w,"\n")
    for x,y in zip(x_data,y_data):
        y_hat = forward(x)
        loss_ral = loss(x,y)
        cost += loss_ral
        number += 1
        print("\t",loss_ral,"\n")
    cost /= number
    print("MSE = ",cost,"\n")
    w_list.append(w)
    MSE_list.append(cost)
plt.plot(w_list,MSE_list)
plt.ylabel("cost")
plt.xlabel("w")
plt.show()

