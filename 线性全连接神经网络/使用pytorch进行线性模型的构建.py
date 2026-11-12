import torch

x_list = [1,2,3,4]
y_list = [2,4,6,8]
w = torch.tensor([1.0])
#开启梯度计算
w.requires_grad = True
a = 0.01

def forward(x):
    return x*w
def loss(x,y):
    y_hat = forward(x)
    return (y_hat-y)**2

print("predict (before training)", 4, forward(4).item())
for epoch in range(100):
    for x,y in zip(x_list,y_list):
        ans = forward(x)
        l = loss(x,y)
        #进行反向传播
        l.backward()
        print("\tgrad:",x,y,w.grad.item())
        #更新权重值
        w.data = w.data - a*w.grad.data
        #将权重值的数值清零
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

print("predict (after training)", 5, forward(5).item())