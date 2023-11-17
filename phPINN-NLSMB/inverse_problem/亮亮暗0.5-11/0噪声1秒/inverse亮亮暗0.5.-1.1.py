"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os

os.environ["DDEBACKEND"] = "tensorflow.compat.v1"
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
import scipy.io  # python读取.mat数据之scipy.io&h5py
import time

start_time = time.time()
# from deepxde.backend import tf


z_lower = -1
z_upper = 1
t_lower = 0
t_upper = 1
# dde.config.set_default_float("float64")
period = 1
A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -40
dpi = 130
nx = 512
nt = 512
x = np.linspace(z_lower, z_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
I = 1j
EExact = 2 * np.exp(-2 * I / 5 * (5 * T - 2 * X)) / np.cosh(2 * T + (22 * X) / 5)
pExact = (
    -I
    / 5
    * np.exp(-2 * I / 5 * (5 * T - 2 * X))
    * (
        (-2 + I) * np.exp(-2 * T - (22 * X) / 5)
        - (2 + I) * np.exp(2 * T + (22 * X) / 5)
    )
    / np.cosh(2 * T + (22 * X) / 5) ** 2
)
etaExact = (5 * np.cosh(2 * T + (22 * X) / 5) ** 2 - 2) / (
    5 * np.cosh(2 * T + (22 * X) / 5) ** 2
)
EExact_u = np.real(EExact)  # (201,256)
EExact_v = np.imag(EExact)
pExact_u = np.real(pExact)
pExact_v = np.imag(pExact)
etaExact_u = np.real(etaExact)  # (201,256)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域

lambda1 = dde.Variable(2.0)  # dtype = float64
lambda2 = dde.Variable(2.0)
omega = dde.Variable(2.0)
True_lambda1 = 0.5  # dtype = float64
True_lambda2 = -1
True_omega = 1


def pde(x, y):  # 这里x其实是x和t，y其实是u和v
    """
    INPUTS:
        x: x[:,0] is z-coordinate
           x[:,1] is t-coordinate
        y: Network output, in this case:
            y[:,0] is u(x,t) the real part
            y[:,1] is v(x,t) the imaginary part
    OUTPUT:
        The pde in standard form i.e. something that must be zero
    """

    Eu = y[:, 0:1]
    Ev = y[:, 1:2]
    pu = y[:, 2:3]
    pv = y[:, 3:4]
    eta = y[:, 4:5]

    # In 'jacobian', i is the output component and j is the input component
    pu_t = dde.grad.jacobian(y, x, i=2, j=1)  # 一阶导用jacobian，二阶导用hessian
    pv_t = dde.grad.jacobian(y, x, i=3, j=1)
    eta_t = dde.grad.jacobian(y, x, i=4, j=1)

    Eu_z = dde.grad.jacobian(y, x, i=0, j=0)  # 一阶导用jacobian，二阶导用hessian
    Ev_z = dde.grad.jacobian(y, x, i=1, j=0)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    Eu_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
    Ev_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)

    f1_u = lambda1 * Eu_tt - lambda2 * Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_z
    f1_v = lambda1 * Ev_tt - lambda2 * Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_z
    f2_u = 2 * Ev * eta - pv_t + 2 * pu * omega
    f2_v = -2 * Eu * eta + pu_t + 2 * pv * omega
    f3 = 2 * pv * Ev + 2 * pu * Eu + eta_t

    return [f1_u, f1_v, f2_u, f2_v, f3]


# x=x[:, 0:1]
# t=x[:, 1:2]**
"""精确解"""
I = 1j


def Eu_func(x):
    return (
        2
        * np.cos(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
        / np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5)
    )


def Ev_func(x):
    return (
        -2
        * np.sin(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
        / np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5)
    )


def pu_func(x):
    return (
        (
            np.cos(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
            + 2 * np.sin(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
        )
        * np.exp(-2 * x[:, 1:2] - (22 * x[:, 0:1]) / 5)
        - np.exp(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5)
        * (
            np.cos(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
            - 2 * np.sin(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
        )
    ) / (5 * np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5) ** 2)


def pv_func(x):
    return (
        (
            2 * np.cos(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
            - np.sin(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
        )
        * np.exp(-2 * x[:, 1:2] - (22 * x[:, 0:1]) / 5)
        + 2
        * (
            np.cos(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5)
            + np.sin(2 * x[:, 1:2] - (4 * x[:, 0:1]) / 5) / 2
        )
        * np.exp(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5)
    ) / (5 * np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5) ** 2)


def eta_func(x):
    return (5 * np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5) ** 2 - 2) / (
        5 * np.cosh(2 * x[:, 1:2] + (22 * x[:, 0:1]) / 5) ** 2
    )


idx = np.random.choice(X_star.shape[0], 2000, replace=False)
X_u_train = X_star[idx, :]  # (2000,2)
EExact_u_1d = Eu_func(X_u_train)
EExact_v_1d = Ev_func(X_u_train)
pExact_u_1d = pu_func(X_u_train)
pExact_v_1d = pv_func(X_u_train)
etaExact_u_1d = eta_func(X_u_train)

# noise = 0.05
# EExact_u_1d = EExact_u_1d + noise*np.std(EExact_u_1d)*np.random.randn(EExact_u_1d.shape[0], EExact_u_1d.shape[1])
# EExact_v_1d = EExact_v_1d + noise*np.std(EExact_v_1d)*np.random.randn(EExact_v_1d.shape[0], EExact_v_1d.shape[1])
# pExact_u_1d = pExact_u_1d + noise*np.std(pExact_u_1d)*np.random.randn(pExact_u_1d.shape[0], pExact_u_1d.shape[1])
# pExact_v_1d = pExact_v_1d + noise*np.std(pExact_v_1d)*np.random.randn(pExact_v_1d.shape[0], pExact_v_1d.shape[1])
# etaExact_u_1d = etaExact_u_1d + noise*np.std(etaExact_u_1d)*np.random.randn(etaExact_u_1d.shape[0], etaExact_u_1d.shape[1])

observe_y = dde.icbc.PointSetBC(X_u_train, EExact_u_1d, component=0)
observe_y1 = dde.icbc.PointSetBC(X_u_train, EExact_v_1d, component=1)
observe_y2 = dde.icbc.PointSetBC(X_u_train, pExact_u_1d, component=2)
observe_y3 = dde.icbc.PointSetBC(X_u_train, pExact_v_1d, component=3)
observe_y4 = dde.icbc.PointSetBC(X_u_train, etaExact_u_1d, component=4)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_y, observe_y1, observe_y2, observe_y3, observe_y4],
    num_domain=10000,
    # num_boundary=20,
    anchors=X_u_train,
    # solution=func,
    # num_test=10000,
)

layer_size = [2] + [64] * 6 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam",
    lr=0.001,
    loss_weights=[1, 1, 1, 1, 1, 100, 100, 100, 100, 100],
    # decay=("inverse time", 5000, 0.6),
    external_trainable_variables=[lambda1, lambda2, omega],
)
filenamevar = "variables1.dat"
variable = dde.callbacks.VariableValue(
    [lambda1, lambda2, omega],
    period=period,
    filename=filenamevar,  # 加了filename就不会打印了
    precision=7,
)
losshistory, train_state = model.train(
    iterations=20000, display_every=100, callbacks=[variable]
)

# dde.optimizers.config.set_LBFGS_options(
#     maxcor=50,
#     ftol=1.0 * np.finfo(float).eps,
#     gtol=1e-08,
#     maxiter=2000,
#     maxfun=None,
#     maxls=50,
# )
model.compile(
    "L-BFGS",
    # loss_weights=[1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100],
    external_trainable_variables=[lambda1, lambda2, omega],
)
losshistory, train_state = model.train(display_every=100, callbacks=[variable])

"""精确解"""
EExact_h = np.abs(EExact)  # （201，256）
Eh_true = EExact_h.flatten()  # (51456,)
pExact_h = np.abs(pExact)
ph_true = pExact_h.flatten()
etaExact_h = np.abs(etaExact)
etah_true = etaExact_h.flatten()
prediction = model.predict(
    X_star, operator=None
)  # 如果 `operator` 为 `None`，则返回网络输出，否则返回 `operator` 的输出
Eu_pred = prediction[:, 0]  # (51456,)
Ev_pred = prediction[:, 1]
Eh_pred = np.sqrt(Eu_pred**2 + Ev_pred**2)
pu_pred = prediction[:, 2]
pv_pred = prediction[:, 3]
ph_pred = np.sqrt(pu_pred**2 + pv_pred**2)
etau_pred = prediction[:, 4]
etah_pred = np.abs(etau_pred)
E_L2_relative_error = dde.metrics.l2_relative_error(Eh_true, Eh_pred)
p_L2_relative_error = dde.metrics.l2_relative_error(ph_true, ph_pred)
eta_L2_relative_error = dde.metrics.l2_relative_error(etah_true, etah_pred)
print("E L2 relative error: %e" % E_L2_relative_error)
print("p L2 relative error: %e" % p_L2_relative_error)
print("eta L2 relative error: %e" % eta_L2_relative_error)
elapsed = time.time() - start_time
"""预测解"""
EH_pred = Eh_pred.reshape(nt, nx)
pH_pred = ph_pred.reshape(nt, nx)
etaH_pred = etah_pred.reshape(nt, nx)
# reopen saved data using callbacks in fnamevar
lines = open(filenamevar, "r").readlines()
# read output data in fnamevar (this line is a long story...)
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

error_lambda_1 = abs((True_lambda1 - Chat[-1, 0]) / True_lambda1) * 100
error_lambda_2 = abs((True_lambda2 - Chat[-1, 1]) / True_lambda2) * 100
error_omega = abs((True_omega - Chat[-1, 2]) / True_omega) * 100

print("Error l1: %.5f%%" % (error_lambda_1))
print("Error l2: %.5f%%" % (error_lambda_2))
print("Error o: %.5f%%" % (error_omega))

fig5 = plt.figure("3d预测演化图E", dpi=dpi)
ax = fig5.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    EH_pred,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="Spectral",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set(xlabel="$z$", ylabel="$t$", zlabel="$|E(t,z)|$")
# fig5.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
plt.tight_layout()  # 自动调整大小和间距，使各个子图标签不重叠

fig6 = plt.figure("3d预测演化图p", dpi=dpi)
ax = fig6.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    pH_pred,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="Spectral",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set(xlabel="$z$", ylabel="$t$", zlabel="$|p(t,z)|$")
# fig6.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
plt.tight_layout()  # 自动调整大小和间距，使各个子图标签不重叠

fig7 = plt.figure("3d预测演化图eta", dpi=dpi)
ax = fig7.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    etaH_pred,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="Spectral",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set(xlabel="$z$", ylabel="$t$", zlabel="$|\eta(t,z)|$")
# fig7.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
plt.tight_layout()  # 自动调整大小和间距，使各个子图标签不重叠

tt0 = 0.1
tt1 = 0.4
fig15 = plt.figure("平面预测演化图", dpi=dpi, constrained_layout=True)
norm0 = matplotlib.colors.Normalize(
    vmin=np.min([EExact_h, pExact_h, etaExact_h, EH_pred, pH_pred, etaH_pred]),
    vmax=np.max([EExact_h, pExact_h, etaExact_h, EH_pred, pH_pred, etaH_pred]),
)
plt.suptitle("Prediction Dynamics")
ax0 = plt.subplot(3, 1, 1)
ax0.set_ylabel("$|E(t,z)|$")
h0 = ax0.imshow(
    EH_pred.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax0)
ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("$|p(t,z)|$")
h1 = ax1.imshow(
    pH_pred.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax1)
ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$|\eta(t,z)|$")
h2 = ax2.imshow(
    etaH_pred.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax2)
fig15.colorbar(h0, ax=[ax0, ax1, ax2], location="right")
ax0.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax0.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax0.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
ax1.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
ax2.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.legend(
    loc="center",
    bbox_to_anchor=(0.6, -0.4),
    borderaxespad=0.0,
    fontsize=10,
    frameon=False,
)  # bbox_to_anchor的坐标决定loc那个点的位置
# plt.subplots_adjust(left=0.15, right=1-0.01,bottom=0.08, top=1-0.08,wspace=None, hspace=0.25)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠

fig16 = plt.figure("平面实际演化图", dpi=dpi, constrained_layout=True)
# norm0 = matplotlib.colors.Normalize(vmin=np.min([EExact_h,pExact_h,etaExact_h,EH_pred,pH_pred,etaH_pred]),vmax=np.max([EExact_h,pExact_h,etaExact_h,EH_pred,pH_pred,etaH_pred]))
plt.suptitle("Exact Dynamics")
ax0 = plt.subplot(3, 1, 1)
ax0.set_ylabel("$|E(t,z)|$")
h0 = ax0.imshow(
    EExact_h.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax0)

ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("$|p(t,z)|$")
h1 = ax1.imshow(
    pExact_h.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax1)

ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$|\eta(t,z)|$")
h2 = ax2.imshow(
    etaExact_h.T,
    interpolation="nearest",
    cmap="jet",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax2)
fig16.colorbar(h0, ax=[ax0, ax1, ax2], location="right")
ax0.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax0.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax0.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
ax1.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
ax2.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax2.legend(
    loc="center",
    bbox_to_anchor=(0.6, -0.4),
    borderaxespad=0.0,
    fontsize=10,
    frameon=False,
)  # bbox_to_anchor的坐标决定loc那个点的位置
# plt.subplots_adjust(left=0.15, right=1-0.01,bottom=0.08, top=1-0.08,wspace=None, hspace=0.25)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠

fig17 = plt.figure("平面绝对误差演化图", dpi=dpi, constrained_layout=True)
norm1 = matplotlib.colors.Normalize(
    vmin=np.min(
        [
            np.abs(EExact_h - EH_pred),
            np.abs(pExact_h - pH_pred),
            np.abs(etaExact_h - etaH_pred),
        ]
    ),
    vmax=np.max(
        [
            np.abs(EExact_h - EH_pred),
            np.abs(pExact_h - pH_pred),
            np.abs(etaExact_h - etaH_pred),
        ]
    ),
)
plt.suptitle("Absolute error")
ax0 = plt.subplot(3, 1, 1)
# ax0.set_title("Absolute error dynamics")
ax0.set_ylabel("$|E(t,z)|$")
h0 = ax0.imshow(
    np.abs(EExact_h - EH_pred).T,
    interpolation="nearest",
    cmap="pink_r",  # PuOr
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax0)
ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("$|p(t,z)|$")
h1 = ax1.imshow(
    np.abs(pExact_h - pH_pred).T,
    interpolation="nearest",
    cmap="pink_r",
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax1)
ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$|\eta(t,z)|$")
h2 = ax2.imshow(
    np.abs(etaExact_h - etaH_pred).T,
    interpolation="nearest",
    cmap="pink_r",  # seismic
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax2)
fig17.colorbar(h0, ax=[ax0, ax1, ax2], location="right")
# plt.subplots_adjust(left=0.15, right=1-0.01,bottom=0.08, top=1-0.08,wspace=None, hspace=0.25)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠
# fnamevar = r"F:\QQ文件\NLS-MB项目\dde\inverse_problem\variables.dat"
fig = plt.figure("反问题系数", dpi=dpi, facecolor=None, edgecolor=None)

l, c = Chat.shape
plt.plot(range(0, period * l, period), Chat[:, 0], "r-")
plt.plot(range(0, period * l, period), Chat[:, 1], "k-")
plt.plot(range(0, period * l, period), Chat[:, 2], "g-")
plt.plot(range(0, period * l, period), np.ones(Chat[:, 0].shape) * True_lambda1, "r--")
plt.plot(range(0, period * l, period), np.ones(Chat[:, 1].shape) * True_lambda2, "k--")
plt.plot(range(0, period * l, period), np.ones(Chat[:, 2].shape) * True_omega, "g--")
plt.legend(
    [
        "$\lambda_1$",
        "$\omega$",
        "$\lambda_2$",
        "True $\lambda_1$",
        "True $\omega$",
        "True $\lambda_2$",
    ]
)
plt.xlabel("iterations")
# plt.tight_layout()


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
scipy.io.savemat(
    "反问题亮亮暗0的噪声0.5,-1,1.mat",
    {
        "x": x,
        "t": t,
        "elapsed": elapsed,
        "X_u_train": X_u_train,
        "E_L2_relative_error": E_L2_relative_error,
        "p_L2_relative_error": p_L2_relative_error,
        "eta_L2_relative_error": eta_L2_relative_error,
        "EH_pred": EH_pred,
        "pH_pred": pH_pred,
        "etaH_pred": etaH_pred,
        "EExact_h": EExact_h,
        "pExact_h": pExact_h,
        "etaExact_h": etaExact_h,
        "error_lambda_1": error_lambda_1,
        "error_lambda_2": error_lambda_2,
        "error_omega": error_omega,
    },
)
plt.show()
