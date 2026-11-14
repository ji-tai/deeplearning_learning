import torch 

# 输入的特征向量
input_size = 4
# 输出的维度，或者是多分类的类别
hidden_size = 4
batch_size = 1

# 1.构建数据集：rnn 的数据集是一个序列类型的数据
idx2char = ['h','e','l','o']
# 这里的xy相当于是一个序列
x_data = [0,1,2,2,3]
y_data = [3,0,2,3,2]

one_hot_lookup = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
y_one_hot = [one_hot_lookup[y] for y in y_data]

inputs = torch.tensor(x_one_hot,dtype=torch.float32).view(-1,1,input_size)
labels = torch.tensor(y_one_hot,dtype=torch.float32).view(-1,1,hidden_size)

# 2.构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size = input_size, hidden_size = hidden_size, batch_size = batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs,hidden)
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.input_size)
rnn_net = Model(input_size,hidden_size,batch_size)

# 3.构建损失函数和更新器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_net.parameters(),lr = 0.1)

# 4. 训练过程
# *****可以深刻的通过训练过程理解rnn的特点，rnn是一种可以灵活处理单个数据或序列数据的一种网络模式，1-多，多-1，多-多
# 损失函数的计算是每一次时间步的损失函数的叠加，，这个例子是输入序列长度与输出序列长度相一致
for epoch in range(40):
    loss = 0
    hidden = rnn_net.init_hidden()
    optimizer.zero_grad()
    print("predicted string ； ", end=" ")
    for input,label in zip(inputs,labels):
        hidden = rnn_net(input,hidden)
        loss += criterion(hidden,label)
        _,idx = hidden.max(dim=1)
        print(idx2char[idx.item()],end=" ")
    loss.backward()
    optimizer.step()
    print(f", Epoch {(epoch+1)} loss = {loss:.4f}")
