import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
import torch.nn.functional as F


class PINN(nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()

        self.activation = activation
        initializer = torch.nn.init.xavier_normal_
        initializer_zero = torch.nn.init.zeros_

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))

        x = self.linears[-1](x)

        return x

    def loss_BC(self, x, y):
        loss_u = F.mse_loss(self.forward(x), y)

        return loss_u

    def pde(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        x.requires_grad = True
        t.requires_grad = True
        X1 = torch.concat([x, t], axis=-1)

        y = self.forward(X1)
        dy_x = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
        dy_t = torch.autograd.grad(
            y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
        dy_xx = torch.autograd.grad(
            dy_x,
            x,
            grad_outputs=torch.ones_like(dy_x),
            create_graph=True,
            retain_graph=True,
        )[0]
        # eq_pred = dy_t + y * dy_x - 0.01 / np.pi * dy_xx

        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    def loss(self, x, y, x_to_train_f):
        loss_u = self.loss_BC(x, y)

        loss_f = torch.mean(torch.square(self.pde(x_to_train_f)))

        loss_val = loss_u + loss_f

        return loss_val


def ic_bcs(XT):
    z = XT[:, 0:1]
    # t = XT[:, 1:2]

    u = -np.sin(np.pi * z)

    return u


layer_sizes = [2] + [32] * 3 + [1]
activation = F.tanh
model = PINN(layer_sizes, activation)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0007)

x_lower = -1
x_upper = 1
t_lower = 0
t_upper = 1

nx = 256
nt = 256
x = np.linspace(x_lower, x_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

ic = X_star[:, 1] == t_lower
idx_ic = np.random.choice(np.where(ic)[0], 200, replace=False)
lb = X_star[:, 0] == x_lower
idx_lb = np.random.choice(np.where(lb)[0], 200, replace=False)
ub = X_star[:, 0] == x_upper
idx_ub = np.random.choice(np.where(ub)[0], 200, replace=False)
icbc_idx = np.hstack((idx_lb, idx_ic, idx_ub))
X_u_train = X_star[icbc_idx]

icbcs_u = ic_bcs(X_u_train)


# 生成20000个点的x坐标和t坐标
idx = np.random.choice(X_star.shape[0], 20000, replace=False)
x_to_train_f = X_star[idx, :]  # (20000,2)

# 生成20000个点的x坐标和y坐标
# x = np.random.uniform(-1, 1, 20000)
# t = np.random.uniform(0, 1, 20000)
# x_to_train_f = np.column_stack((x, t))

X_u_train = torch.from_numpy(X_u_train).float()
icbcs_u = torch.from_numpy(icbcs_u).float()
x_to_train_f = torch.from_numpy(x_to_train_f).float()

epoch = 1
while epoch <= 6000:
    # Pred = model(Xb, Xt)
    loss = model.loss(X_u_train, icbcs_u, x_to_train_f)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 500 == 0:
        print("epoch:", epoch, "loss:", float(loss))
    epoch += 1

pred = model(torch.from_numpy(X_star).float()).detach().cpu().numpy()
f = model.pde(torch.from_numpy(X_star).float()).detach().cpu().numpy()
print("Mean residual:", np.mean(np.absolute(f)))


def gen_testdata():
    data = np.load(r"Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


X, y_true = gen_testdata()
y_pred = model(torch.from_numpy(X).float()).detach().cpu().numpy()
f = model.pde(torch.from_numpy(X).float()).detach().cpu().numpy()
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
plt.figure()
plt.title("true")
plt.tricontourf(X[:, 0], X[:, 1], y_true[:, 0], levels=256, cmap="jet")
plt.colorbar()
plt.savefig("真解.png")

plt.figure()
plt.title("pred")
plt.tricontourf(X[:, 0], X[:, 1], y_pred[:, 0], levels=256, cmap="jet")
plt.colorbar()
plt.savefig("预测解.png")
plt.show()
