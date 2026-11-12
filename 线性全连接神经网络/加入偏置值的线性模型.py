import torch 

x_list = [1,2,3,4]
y_list = [2,4,6,8]
#设置两个参数
w1 = torch.tensor([1.0])
w2 = torch.tensor([2.0])
#设置偏置值
b = torch.tensor([1.0])
#设置学习率
a = 0.001
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

#模型函数
def forward(x):
    y_hat = w1*x*x+w2*x+b
    return y_hat
#损失函数
def loss(x,y):
    y_hat = w1*x*x+w2*x+b
    return (y_hat-y)**2
print("predict (before training)", 5, forward(5).item())
for epoch in range(100):
    for x,y in zip(x_list,y_list):
        ans_ral = forward(x)
        loss_ral = loss(x,y)
        #反向传播
        loss_ral.backward()
        print("\tx:",x,"y:",y,"w1.grad:",w1.grad.item(),"w2.grad:",w2.grad.item(),"b.grad:",b.grad.item())
        w1.data = w1.data - a*w1.grad.data
        w2.data = w2.data - a*w2.grad.data
        b.data = b.data - a*b.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, loss_ral.item())
print("predict (after training)", 5, forward(5).item())






