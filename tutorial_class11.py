import torch
import torch.nn as nn


def use_rnncell():
    batch_size = 1
    seq_len = 3
    # 假如我们有id = 1,2,3,4,5,6,7,8,9,10一共10个sample。
    # 假设我们设定seq_len是3。
    # 那现在数据的形式应该为1-2-3，2-3-4，3-4-5，4-5-6，5-6-7，6-7-8，7-8-9，8-9-10，9-10-0，10-0-0（最后两个数据不完整，进行补零）的10个数据。
    # 假设我们设定batch_size为2。
    # 那我们取出第一个batch为1-2-3，2-3-4。这个batch的size就是(2，3，feature_dims)了。我们把这个玩意儿喂进模型。
    # 接下来第二个batch为3-4-5，4-5-6。
    # 第三个batch为5-6-7，6-7-8。
    # 第四个batch为7-8-9，8-9-10。
    # 第五个batch为9-10-0，10-0-0。我们的数据一共生成了5个batch
    input_size = 4
    # 是指输入有多少种字符，而不是多少个字符
    hidden_size = 2

    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    dataset = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(batch_size, hidden_size)

    for idx, data in enumerate(dataset):
        # idx指序列
        print('=' * 20, idx, '=' * 20)
        print('Input size:', data.shape, data)

        hidden = cell(data, hidden)

        print('hidden size:', hidden.shape, hidden)
        print(hidden)


def use_rnn():
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1

    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    inputs = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    out, hidden = cell(inputs, hidden)

    print('Output size:', out.shape)  # (seq_len, batch_size, hidden_size)
    print('Output:', out)
    print('Hidden size:', hidden.shape)  # (num_layers, batch_size, hidden_size)
    print('Hidden:', hidden)


def example_rnncell():
    batch_size = 1
    input_size = 4
    hidden_size = 4

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]  # hello中各个字符的下标
    y_data = [3, 1, 2, 3, 2]  # ohlol中各个字符的下标

    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]
    x_one_hot = [one_hot_lookup[x] for x in x_data]  # (seqLen, inputSize)

    inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
    labels = torch.LongTensor(y_data).view(-1, 1)
    print(inputs.shape, labels.shape)

    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, batch_size):
            super(Model, self).__init__()
            self.batch_size = batch_size
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        def forward(self, inputs, hidden):
            hidden = self.rnncell(inputs, hidden)
            return hidden

        def init_hidden(self):
            return torch.zeros(self.batch_size, self.hidden_size)

    net = Model(input_size, hidden_size, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(15):
        loss = 0
        optimizer.zero_grad()
        hidden = net.init_hidden()
        print('Predicted string:', end='')
        for input, label in zip(inputs, labels):
            hidden = net(input, hidden)
            # 注意交叉熵在计算loss的时候维度关系，这里的hidden是([1, 4]), label是 ([1])
            loss += criterion(hidden, label)
            _, idx = hidden.max(dim=1)
            print(idx2char[idx.item()], end='')

        loss.backward()
        optimizer.step()
        print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))


def example_rnn():
    batch_size = 1
    input_size = 4
    hidden_size = 4
    seq_len = 5
    num_layers = 1

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]  # hello中各个字符的下标
    y_data = [3, 1, 2, 3, 2]  # ohlol中各个字符的下标

    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]
    x_one_hot = [one_hot_lookup[x] for x in x_data]  # (seqLen, inputSize)

    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    labels = torch.LongTensor(y_data)
    print(inputs.shape, labels.shape)

    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
            super(Model, self).__init__()
            self.num_layers = num_layers
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        def forward(self, inputs):
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            out, _ = self.rnn(inputs, hidden)  # 注意输出维度是(seqLen, batch_size, hidden_size)
            return out.view(-1, self.hidden_size)  # 为了容易计算交叉熵这里调整维度为(seqLen * batch_size, hidden_size)

    net = Model(input_size, hidden_size, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


def emb():
    # parameters
    num_class = 4
    input_size = 4
    hidden_size = 8
    embedding_size = 10
    num_layers = 2
    batch_size = 1
    seq_len = 5

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb = nn.Embedding(input_size, embedding_size)
            self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_class)

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            x = self.emb(x)  # (batch, seqLen, embeddingSize)
            x, _ = self.rnn(x, hidden)  # 输出(𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 𝒔𝒆𝒒𝑳𝒆𝒏, hidden_size)
            x = self.fc(x)  # 输出(𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 𝒔𝒆𝒒𝑳𝒆𝒏, 𝒏𝒖𝒎𝑪𝒍𝒂𝒔𝒔)
            return x.view(-1, num_class)  # reshape to use Cross Entropy: (𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆×𝒔𝒆𝒒𝑳𝒆𝒏, 𝒏𝒖𝒎𝑪𝒍𝒂𝒔𝒔)


    net = Model()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
    y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)

    inputs = torch.LongTensor(x_data)
    labels = torch.LongTensor(y_data)

    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


if __name__ == '__main__':
    # use_rnncell()
    # use_rnn()
    # example_rnncell()
    # example_rnn()
    emb()