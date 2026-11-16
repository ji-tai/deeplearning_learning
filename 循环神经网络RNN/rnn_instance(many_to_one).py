import torch
# 1. 构建数据集

input_size = 4
hidden_size = 5
batch_size = 1
output_size = 2
seq_size = 3


# 词汇表
idx2word = ['this', 'movie', 'good', 'bad']
predict2word = ["bad","good"]

# 两句话
# this movie good -> 正面
# this movie bad  -> 负面

x_data = [
    [0, 1, 2],
    [0, 1, 3]
]

y_data = [1, 0]

one_hot_lookup = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

x_one_hot = [[one_hot_lookup[word] for word in sentence]for sentence in x_data]# [2_sentence,3_word,4_input_size]

inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(-1,seq_size,batch_size,input_size)
labels = torch.tensor(y_data, dtype=torch.long)# [2]


# 2. 构建模型

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.rnncell = torch.nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.fc = torch.nn.Linear(
            hidden_size, output_size
        )

    def forward(self, inputs, hidden):

        # 逐个处理输入序列
        for input in inputs:# 取每一个句子的单词
            hidden = self.rnncell(input, hidden)

        # 只使用最后一个隐藏状态进行分类
        output = self.fc(hidden)

        return output

    def init_hidden(self):
        return torch.zeros(batch_size, hidden_size)# [1,5]


rnn_net = Model()


# 3. 构建损失函数和优化器

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_net.parameters(), lr=0.1)


# 4. 训练过程

for epoch in range(100):

    loss = 0

    optimizer.zero_grad()

    for input, label in zip(inputs, labels):# 取第一个句子，和第一个标签
        hidden = rnn_net.init_hidden()
        output = rnn_net(input, hidden)
        loss += criterion(output,label.view(1))
        
    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch+1}, loss = {loss.item():.4f}"
    )


# 5. 测试

with torch.no_grad():

    for input in inputs:

        hidden = rnn_net.init_hidden()

        output = rnn_net(input, hidden)

        _, predicted = output.max(dim=1)
        _, idx = input.max(dim=2)
        words = [idx2word[i.item()] for i in idx]
        s = " ".join(words)
        print("题目：",s)
        print("预测类别:", predict2word[predicted.item()])
