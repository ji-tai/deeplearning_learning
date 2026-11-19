# many_to_many(不等长)
# 输入：a b
# 输出：b c a
# torch.nn.CrossEntropyLoss()的数据类型
# 1. 使用类别索引编码
# 变量	    Shape	数据类型
# output	(1, 4)	float32
# target	(1,)	int64（long）
# 2. 使用独热向量编码
# 变量	Shape	数据类型
# hidden	(1, 4)	float32
# label	(1, 4)	float32

import torch

# 1. 构建数据集

input_size = 4
hidden_size = 8
output_size = 4
batch_size = 1

idx2char = ['a', 'b', 'c', '<SOS>']

# 输入序列长度为2
x_data = [0, 1]

# 输出序列长度为3
y_data = [1, 2, 0]

# 独热编码表
one_hot_lookup = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

inputs = torch.tensor([one_hot_lookup[x] for x in x_data],dtype=torch.float32).view(-1, batch_size, input_size)# (seq_len_2,batch_size_1,input_size_4)

labels = torch.tensor(y_data, dtype=torch.long)

# 2. 构建模型

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.RNNCell(input_size, hidden_size)

        # Decoder
        self.decoder = torch.nn.RNNCell(output_size, hidden_size)

        # 输出层
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden, targets):

        # Encoder：读取完整输入序列
        for input in inputs:
            hidden = self.encoder(input, hidden)
        # 将所有的信息压缩到最后一个序列
        # Decoder：从起始标记开始
        decoder_input = torch.tensor([[0., 0., 0., 1.]])# 目的是创建一个浮点型的数据列表

        loss = 0

        for target in targets:# 这里的targets相当于是目标序列
            # 此时输入的hidden是经过encode的编码过后获得的富含语义的向量
            hidden = self.decoder(decoder_input, hidden)
            output = self.fc(hidden)
            loss += criterion(output, target.view(1))
            # 使用真实字符作为下一步输入
            decoder_input = torch.tensor([one_hot_lookup[target.item()]],dtype=torch.float32)
        return loss

    def init_hidden(self):
        return torch.zeros(batch_size, hidden_size)


rnn_net = Model()


# 3. 构建损失函数和优化器

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    rnn_net.parameters(), lr=0.05
)


# 4. 训练过程

for epoch in range(200):

    hidden = rnn_net.init_hidden()

    optimizer.zero_grad()

    loss = rnn_net(inputs, hidden, labels)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}, loss = {loss.item():.4f}"
        )


# 5. 测试：自回归生成

with torch.no_grad():

    hidden = rnn_net.init_hidden()

    # Encoder
    for input in inputs:
        hidden = rnn_net.encoder(input, hidden)

    # Decoder
    decoder_input = torch.tensor(
        [[0., 0., 0., 1.]]
    )

    result = []

    for _ in range(3):

        hidden = rnn_net.decoder(
            decoder_input, hidden
        )

        output = rnn_net.fc(hidden)

        _, idx = output.max(dim=1)

        result.append(idx2char[idx.item()])

        decoder_input = torch.tensor(
            [one_hot_lookup[idx.item()]],
            dtype=torch.float32
        )

    print("翻译结果:", result)
