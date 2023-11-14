"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os
os.environ["DDEBACKEND"] = "pytorch"
import numpy as np

import deepxde as dde

# For plotting
import matplotlib.pyplot as plt

x_lower = -2
x_upper = 2
t_lower = -2
t_upper = 2
nx = 256
nt = 201
# Creation of the 2D domain (for plotting and input)
x = np.linspace(x_lower, x_upper, nx)
t = np.linspace(t_lower, t_upper, nt)
X, T = np.meshgrid(x, t)

# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(x_lower, x_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域


# The "physics-informed" part of the loss
def pde(x, y):
    """
    INPUTS:
        x: x[:,0] is x-coordinate
           x[:,1] is t-coordinate
        y: Network output, in this case:
            y[:,0] is u(x,t) the real part
            y[:,1] is v(x,t) the imaginary part
    OUTPUT:
        The pde in standard form i.e. something that must be zero
    """

    u = y[:, 0:1]
    v = y[:, 1:2]

    # In 'jacobian', i is the output component and j is the input component
    u_t = dde.grad.jacobian(y, x, i=0, j=1)  # 一阶导用jacobian，二阶导用hessian
    v_t = dde.grad.jacobian(y, x, i=1, j=1)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

    f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
    f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

    return [f_u, f_v]


# x=x[:, 0:1]
# t=x[:, 1:2]
# Initial conditions
def icbc_u(x):
    return (
        8 * np.sin(x[:, 1:2]) * x[:, 1:2]
        + (4 * x[:, 1:2] ** 2 + 4 * x[:, 0:1] ** 2 - 3) * np.cos(x[:, 1:2])
    ) / (4 * x[:, 1:2] ** 2 + 4 * x[:, 0:1] ** 2 + 1)


def icbc_v(x):
    return (
        (4 * x[:, 1:2] ** 2 + 4 * x[:, 0:1] ** 2 - 3) * np.sin(x[:, 1:2])
        - 8 * np.cos(x[:, 1:2]) * x[:, 1:2]
    ) / (4 * x[:, 1:2] ** 2 + 4 * x[:, 0:1] ** 2 + 1)


# (1 - 4*(1 + 2*I*t)/(4*t^2 + 4*x^2 + 1))*exp(t*I)
bc_u = dde.icbc.DirichletBC(
    geomtime, icbc_u, lambda _, on_boundary: on_boundary, component=0
)
bc_v = dde.icbc.DirichletBC(
    geomtime, icbc_v, lambda _, on_boundary: on_boundary, component=1
)
ic_u = dde.icbc.IC(
    geomtime, icbc_u, lambda _, on_initial: on_initial, component=0
)  # 所有初始条件上的点都应用初始条件
ic_v = dde.icbc.IC(
    geomtime, icbc_v, lambda _, on_initial: on_initial, component=1
)  # 所有初始条件上的点都应用初始条件
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_u, bc_v, ic_u, ic_v],
    num_domain=20000,
    num_boundary=100,
    num_initial=50,
    train_distribution="pseudo",
)

# Network architecture
net = dde.nn.FNN([2] + [64] * 4 + [2], "tanh", "Glorot normal")
# net = dde.nn.PFNN([2, [32, 32], [32, 32],[32, 32],[32, 32], 2], "tanh", "Glorot uniform") #子网

model = dde.Model(data, net)
# To employ a GPU accelerated system is highly encouraged.

model.compile("adam", lr=0.001, loss="MSE")
losshistory, train_state = model.train(iterations=20000, display_every=100)

dde.optimizers.config.set_LBFGS_options(
    maxcor=50,
    ftol=1.0 * np.finfo(float).eps,
    gtol=1e-08,
    maxiter=5000,
    maxfun=None,
    maxls=50,
)
model.compile("L-BFGS")
model.train(display_every=100)

# Make prediction
prediction = model.predict(X_star, operator=None)

u = prediction[:, 0].reshape(nt, nx)
v = prediction[:, 1].reshape(nt, nx)

h = np.sqrt(u**2 + v**2)

# Plot predictions
fig, ax = plt.subplots(3)

ax[0].set_title("Results")
ax[0].set_ylabel("Real part")
ax[0].imshow(
    u.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)
ax[1].set_ylabel("Imaginary part")
ax[1].imshow(
    v.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)
ax[2].set_ylabel("Amplitude")
ax[2].imshow(
    h.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)

fig5 = plt.figure("预测演化图", dpi=100, facecolor=None, edgecolor=None)
ax = fig5.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    h,
    # rstride=1,  # 指定行的跨度
    # cstride=1,  # 指定列的跨度
    cmap="jet",  # 设置颜色映射（这里是绿化的意思）
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|h(x,t)|$")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

plt.show()
