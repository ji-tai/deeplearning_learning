import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1,2,3,4]
y_data = [2,4,6,8]

#定义模型函数
def forward(x):
    return x*w + b

#定义损失函数
def loss(x,y):
    y_hat = forward(x)
    return (y_hat-y)*(y_hat-y)

w_vals = np.arange(0.0, 4.0, 0.1)   # [0.0, 0.1, 0.2, ..., 3.9]，共40个
b_vals = np.arange(-1.0, 1.0, 0.1)  # [-1.0, -0.9, ..., 0.9]，共20个
W, B = np.meshgrid(w_vals, b_vals)
MSE = np.zeros_like(W)

i = 0

#通过穷举法不断尝试权重值
for w in np.arange(0.0,4.0,0.1):
    j=0
    for b in np.arange(-1.0,1.0,0.1):
        cost = 0
        print("*"*20,"\n")
        print("W = ",w,"\n")
        print("B = ",b,"\n")
        for x,y in zip(x_data,y_data):
            y_hat = forward(x)
            loss_ral = loss(x,y)
            cost += loss_ral
            print("\t",loss_ral,"\n")
        cost /= 4
        MSE[j,i] = cost 
        print("MSE = ",cost,"\n")
        j+=1
    i+=1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, MSE, cmap='viridis', edgecolor='none')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
fig.colorbar(surf)
plt.show()


