'''
基于rnn模型生成的一个简单的教学模型，使得通过训练将输入<SOS>自动自回归生成一个序列阿，a,b,c。
'''
import torch

# 1. 构建数据集

input_size = 4
hidden_size = 8
output_size = 4# 对照输出词表
batch_size = 1

# 词汇表
idx2char = ['a', 'b', 'c', '<SOS>']

# 输入一个起始标记
x_data = [3]

# 需要生成的输出序列
y_data = [0, 1, 2]

one_hot_lookup = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

inputs = torch.tensor([one_hot_lookup[x] for x in x_data],dtype=torch.float32).view(-1, batch_size, input_size)# [1_seq_len,1,4_input_size]

labels = torch.tensor(y_data, dtype=torch.long)


# 2. 构建模型

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.RNNCell(input_size, hidden_size)
        self.decoder = torch.nn.RNNCell(output_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden, targets):
        # Encoder：处理一个输入
        for input in inputs:# 从一个序列中取一个Token(由于是one_to_many,所以seq_len为0)
            # 只处理一次相当于是线性层
            hidden = self.encoder(input, hidden)

        # Decoder：逐步生成输出
        # 这个数据是解码器第一次输入的序列
        decoder_input = torch.tensor([[0., 0., 0., 1.]])# float32数据类型
        loss = 0
        outputs = []

        for target in targets:

            hidden = self.decoder(decoder_input, hidden)

            output = self.fc(hidden)

            loss += criterion(output,target.view(1))# 损失函数

            outputs.append(output)

            # Teacher Forcing：
            # 将真实的上一个字符作为下一个输入
            decoder_input = torch.tensor(
                [one_hot_lookup[target.item()]],
                dtype=torch.float32
            )

        return loss, outputs

    def init_hidden(self):
        return torch.zeros(batch_size, hidden_size)


rnn_net = Model()


# 3. 损失函数和优化器

# output形状	        target形状	                含义
# [N, C]	            [N] 	             N 个样本，每个样本 1 个类别索引
# [N, C, d1, d2, ...]	[N, d1, d2, ...]	分割等任务的逐像素分类
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(rnn_net.parameters(), lr=0.05)


# 4. 训练过程

for epoch in range(200):

    hidden = rnn_net.init_hidden()

    optimizer.zero_grad()

    loss, outputs = rnn_net(inputs, hidden, labels)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}, loss = {loss.item():.4f}"
        )


# 5. 测试：自回归生成

with torch.no_grad():

    hidden = rnn_net.init_hidden()

    for input in inputs:
        hidden = rnn_net.encoder(input, hidden)

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

    print("生成结果:", result)
