import matplotlib.pyplot as plt
import numpy as np
import torch


class FNN(torch.nn.Module):  # 继承torch.nn.Module
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = torch.nn.functional.relu(linear(x))
        x = self.linears[-1](x)
        return x


# Load data
train_data = np.loadtxt("train.txt").astype(np.float32)
train_x, train_y = train_data[:, :1], train_data[:, 1:]
test_data = np.loadtxt("test.txt").astype(np.float32)
test_x, test_y = test_data[:, :1], test_data[:, 1:]

# Hyperparameters
layer_sizes = [1] + [128] * 4 + [1]
lr = 0.001
nsteps = 10000

# Build NN
nn = FNN(layer_sizes)

# Optimizer
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

# Train
for i in range(nsteps):
    y = nn(torch.from_numpy(train_x))  # 手动将numpy转成tensor
    # Loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y, torch.from_numpy(train_y))  # 预测值和真值
    optimizer.zero_grad()  # 清零所有参数的梯度，以便进行下一轮反向传播之前不累积梯度。
    loss.backward()  # 反向传播，计算损失函数关于各个参数的梯度。
    optimizer.step()  # 根据参数的梯度更新参数值，使损失函数尽可能减小。

    if i % 1000 == 0 or i == nsteps - 1:
        with torch.no_grad():  # 进入一个上下文管理器，表示在该代码块中不需要计算梯度。
            pred_y = nn(torch.from_numpy(test_x)).detach().cpu().numpy()
        err_test = np.mean((pred_y - test_y) ** 2)
        print(i, loss.item(), err_test)

# Plot
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()
