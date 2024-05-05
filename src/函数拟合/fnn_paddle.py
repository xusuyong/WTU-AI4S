import paddle
import matplotlib.pyplot as plt
import numpy as np


class FNN(paddle.nn.Layer):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.linears = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = paddle.nn.functional.relu(linear(x))
        x = self.linears[-1](x)
        return x


train_data = np.loadtxt("train.txt").astype(np.float32)
train_x, train_y = train_data[:, :1], train_data[:, 1:]
test_data = np.loadtxt("test.txt").astype(np.float32)
test_x, test_y = test_data[:, :1], test_data[:, 1:]
layer_sizes = [1] + [128] * 4 + [1]
lr = 0.001
nsteps = 10000
nn = FNN(layer_sizes)
optimizer = paddle.optimizer.Adam(
    parameters=nn.parameters(), learning_rate=lr, weight_decay=0.0
)
for i in range(nsteps):
    y = nn(paddle.to_tensor(train_x))
    loss_fn = paddle.nn.MSELoss()
    loss = loss_fn(y, paddle.to_tensor(train_y))
    optimizer.clear_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0 or i == nsteps - 1:
        with paddle.no_grad():
            pred_y = nn(paddle.to_tensor(test_x)).detach().cpu().numpy()
        err_test = np.mean((pred_y - test_y) ** 2)
        print(i, loss.item(), err_test)
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()
