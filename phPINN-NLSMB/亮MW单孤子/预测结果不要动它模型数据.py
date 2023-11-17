import numpy as np

import deepxde as dde

# For plotting
import scipy.io  # python读取.mat数据之scipy.io&h5py
import matplotlib.pyplot as plt

z_lower = -2
z_upper = 2
t_lower = -3
t_upper = 3
"""eta振幅是W型孤子？？？   记住记录L2误差和训练时间、次数等"""
nx = 512
nt = 512
# Creation of the 2D domain (for plotting and input)
x = np.linspace(z_lower, z_upper, nx)
t = np.linspace(t_lower, t_upper, nt)
X, T = np.meshgrid(x, t)
# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

"""用来算L2误差的坐标"""
# nx1 = 256; nt1 = 201
# # Creation of the 2D domain (for plotting and input)
# x1 = np.linspace(z_lower, z_upper, nx1)
# t1 = np.linspace(t_lower, t_upper, nt1)
# X1, T1 = np.meshgrid(x1, t1)
# X_star1 = np.hstack((X1.flatten()[:, None], T1.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域

# The "physics-informed" part of the loss

"""the PDE residual"""


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
    # Eu_t = dde.grad.jacobian(y, x, i=0, j=1)#一阶导用jacobian，二阶导用hessian
    # Ev_t = dde.grad.jacobian(y, x, i=1, j=1)
    pu_t = dde.grad.jacobian(y, x, i=2, j=1)  # 一阶导用jacobian，二阶导用hessian
    pv_t = dde.grad.jacobian(y, x, i=3, j=1)
    eta_t = dde.grad.jacobian(y, x, i=4, j=1)

    Eu_z = dde.grad.jacobian(y, x, i=0, j=0)  # 一阶导用jacobian，二阶导用hessian
    Ev_z = dde.grad.jacobian(y, x, i=1, j=0)
    # pu_z = dde.grad.jacobian(y, x, i=2, j=0)  # 一阶导用jacobian，二阶导用hessian
    # pv_z = dde.grad.jacobian(y, x, i=3, j=0)
    # eta_z = dde.grad.jacobian(y, x, i=4, j=0)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    Eu_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
    Ev_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)

    omega = -1

    f1_u = 0.5 * Eu_tt + Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_z
    f1_v = 0.5 * Ev_tt + Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_z
    f2_u = 2 * Ev * eta - pv_t + 2 * pu * omega
    f2_v = -2 * Eu * eta + pu_t + 2 * pv * omega
    f3 = 2 * pv * Ev + 2 * pu * Eu + eta_t

    # return f1_u + f1_v + f2_u + f2_v + f3
    return [f1_u, f1_v, f2_u, f2_v, f3]


# Boundary and Initial conditions
# Initial conditions
# def sech(x):
#     """sech函数"""
#     return 2 / (np.exp(x) + np.exp(-x))
"""第一列是z，第二列是t"""


# x=x[:, 0:1]
# t=x[:, 1:2]
def Eu_func(x):
    return 2 * np.cos(2 * x[:, 1:2]) / np.cosh(2 * x[:, 1:2] + 6 * x[:, 0:1])


def Ev_func(x):
    return -2 * np.sin(2 * x[:, 1:2]) / np.cosh(2 * x[:, 1:2] + 6 * x[:, 0:1])


def pu_func(x):
    return (
        (np.exp(-2 * x[:, 1:2] - 6 * x[:, 0:1]) - np.exp(2 * x[:, 1:2] + 6 * x[:, 0:1]))
        * np.cos(2 * x[:, 1:2])
        / np.cosh(2 * x[:, 1:2] + 6 * x[:, 0:1]) ** 2
    )


def pv_func(x):
    return (
        -(
            np.exp(-2 * x[:, 1:2] - 6 * x[:, 0:1])
            - np.exp(2 * x[:, 1:2] + 6 * x[:, 0:1])
        )
        * np.sin(2 * x[:, 1:2])
        / np.cosh(2 * x[:, 1:2] + 6 * x[:, 0:1]) ** 2
    )


def eta_func(x):
    return (np.cosh(2 * x[:, 1:2] + 6 * x[:, 0:1]) ** 2 - 2) / np.cosh(
        2 * x[:, 1:2] + 6 * x[:, 0:1]
    ) ** 2


# Boundary conditions,component的意思是神经网络的输出
bc_Eu = dde.icbc.DirichletBC(
    geomtime, Eu_func, lambda _, on_boundary: on_boundary, component=0
)
bc_Ev = dde.icbc.DirichletBC(
    geomtime, Ev_func, lambda _, on_boundary: on_boundary, component=1
)
bc_pu = dde.icbc.DirichletBC(
    geomtime, pu_func, lambda _, on_boundary: on_boundary, component=2
)
bc_pv = dde.icbc.DirichletBC(
    geomtime, pv_func, lambda _, on_boundary: on_boundary, component=3
)
bc_eta = dde.icbc.DirichletBC(
    geomtime, eta_func, lambda _, on_boundary: on_boundary, component=4
)

ic_Eu = dde.icbc.IC(
    geomtime, Eu_func, lambda _, on_initial: on_initial, component=0
)  # 所有初始条件上的点都应用初始条件
ic_Ev = dde.icbc.IC(
    geomtime, Ev_func, lambda _, on_initial: on_initial, component=1
)  # 所有初始条件上的点都应用初始条件
ic_pu = dde.icbc.IC(
    geomtime, pu_func, lambda _, on_initial: on_initial, component=2
)  # 所有初始条件上的点都应用初始条件
ic_pv = dde.icbc.IC(
    geomtime, pv_func, lambda _, on_initial: on_initial, component=3
)  # 所有初始条件上的点都应用初始条件
ic_eta = dde.icbc.IC(
    geomtime, eta_func, lambda _, on_initial: on_initial, component=4
)  # 所有初始条件上的点都应用初始条件
"""初始条件就是t等于-5的时候"""
# 初始条件满足init_cond_u或init_cond_v函数
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_Eu, bc_Ev, bc_pu, bc_pv, bc_eta, ic_Eu, ic_Ev, ic_pu, ic_pv, ic_eta],
    num_domain=25000,
    num_boundary=200,
    num_initial=100,
    train_distribution="pseudo",
)  # 在内部取10000个点，在边界取20个点，在初始取200个点,"pseudo" (pseudorandom)伪随机分布
# 前面输出pde的loss，后面输出初始、边界的loss
# Network architecture
net = dde.nn.FNN([2] + [128] * 6 + [5], "tanh", "Glorot normal")

model = dde.Model(data, net)

resampler = dde.callbacks.PDEPointResampler(period=5000)

model.compile(
    "adam",
    lr=0.001,
    loss="MSE",
    decay=("inverse time", 5000, 0.5),
    # loss_weights=[1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
)
losshistory, train_state = model.train(
    iterations=30000, display_every=100, callbacks=[resampler]
)

# dde.optimizers.config.set_LBFGS_options(
#     maxcor=50,
#     ftol=1.0 * np.finfo(float).eps,
#     gtol=1e-08,
#     maxiter=50000,
#     maxfun=50000,
#     maxls=50,
# )
model.compile(
    "L-BFGS",
    # loss_weights=[1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
)
losshistory, train_state = model.train(display_every=100, callbacks=[resampler])

"""RAR"""
# for i in range(10):#一下添加几个点，总共这些次
#     XTrar = geomtime.random_points(100000)
#     f = model.predict(XTrar, operator=pde)#[0]
#     err_eq = np.absolute(f)[:,-1]
#     err = np.mean(err_eq)
#     print("Mean residual: %.3e" % (err))
#     # print(np.shape(err_eq), err_eq)
#
#     # err_eq = torch.tensor(err_eq)
#     # x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()#返回最大的前十个
#     x_ids = np.argsort(err_eq)[-40:]
#     # print(x_ids)
#
#     for elem in x_ids:
#         print("Adding new point:", XTrar[elem], "\n")
#         data.add_anchors(XTrar[elem])
#     early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
#     # exit()
#     model.compile("adam", lr=0.0005)
#
#     losshistory, train_state = model.train(
#         iterations=2000,display_every=100,disregard_previous_best=True, callbacks=[early_stopping] )
#
#     model.compile("L-BFGS-B")
#     losshistory, train_state = model.train()

"""用来计算L2误差"""
# X_star = geomtime.uniform_points(50000, boundary=True)
"""精确解"""
Eu_true = Eu_func(X_star)[:, 0]
Ev_true = Ev_func(X_star)[:, 0]
Eh_true = np.sqrt(Eu_true**2 + Ev_true**2)
pu_true = pu_func(X_star)[:, 0]
pv_true = pv_func(X_star)[:, 0]
ph_true = np.sqrt(pu_true**2 + pv_true**2)
etau_true = eta_func(X_star)[:, 0]
etah_true = np.abs(etau_true)
# Make prediction
"""预测解"""
prediction = model.predict(
    X_star, operator=None
)  # 如果 `operator` 为 `None`，则返回网络输出，否则返回 `operator` 的输出
Eu_pred = prediction[:, 0]
Ev_pred = prediction[:, 1]
Eh_pred = np.sqrt(Eu_pred**2 + Ev_pred**2)
pu_pred = prediction[:, 2]
pv_pred = prediction[:, 3]
ph_pred = np.sqrt(pu_pred**2 + pv_pred**2)
etau_pred = prediction[:, 4]
etah_pred = np.abs(etau_pred)
print("E L2 relative error: %e" % (dde.metrics.l2_relative_error(Eh_true, Eh_pred)))
print("p L2 relative error: %e" % (dde.metrics.l2_relative_error(ph_true, ph_pred)))
print(
    "eta L2 relative error: %e" % (dde.metrics.l2_relative_error(etah_true, etah_pred))
)

"""精确解"""
EExact_h = griddata(X_star, Eh_true, (X, T), method="cubic")
pExact_h = griddata(X_star, ph_true, (X, T), method="cubic")
etaExact_h = griddata(X_star, etah_true, (X, T), method="cubic")
"""预测解"""
EH_pred = griddata(X_star, Eh_pred, (X, T), method="cubic")
pH_pred = griddata(X_star, ph_pred, (X, T), method="cubic")
etaH_pred = griddata(X_star, etah_pred, (X, T), method="cubic")

# Plot predictions
A = t_upper - t_lower
stride = 1
elevation = 20
azimuth = -40
dpi = 130

fig101 = plt.figure("E对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
t = -2
index = round(
    (t - t_lower) / A * (nt - 1)
)  # index只能是0-200（总共有201行,=0时索引第1个数,=200时索引第201）
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
ax = plt.subplot(2, 1, 2)
t = 2
index = round((t - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
fig101.tight_layout()

fig102 = plt.figure("p对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
t = -2
index = round((t - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
ax = plt.subplot(2, 1, 2)
t = 2
index = round((t - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
fig102.tight_layout()

fig103 = plt.figure("eta对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
t = -2
index = round((t - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
ax = plt.subplot(2, 1, 2)
t = 2
index = round((t - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % t)
plt.legend()
fig103.tight_layout()

fig5 = plt.figure("3d预测演化图E", dpi=dpi, facecolor=None, edgecolor=None)
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
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|E(t,z)|$")
# fig5.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

fig6 = plt.figure("3d预测演化图p", dpi=dpi, facecolor=None, edgecolor=None)
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
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|p(t,z)|$")
# fig6.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

fig7 = plt.figure("3d预测演化图eta", dpi=dpi, facecolor=None, edgecolor=None)
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
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|\eta(t,z)|$")
# fig7.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

fig8 = plt.figure("3d真解E", dpi=dpi, facecolor=None, edgecolor=None)
ax = fig8.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    EExact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="coolwarm",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|E(t,z)|$")
# fig8.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

fig9 = plt.figure("3d真解p", dpi=dpi, facecolor=None, edgecolor=None)
ax = fig9.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    pExact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="coolwarm",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|p(t,z)|$")
# fig9.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

fig10 = plt.figure("3d真解eta", dpi=dpi, facecolor=None, edgecolor=None)
ax = fig10.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    etaExact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="coolwarm",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax.grid(False)#关闭背景的网格线
ax.set_xlabel("$z$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|\eta(t,z)|$")
# fig10.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)

# dde薛定谔里的图
fig15 = plt.figure("平面预测演化图", dpi=dpi)
ax0 = plt.subplot(3, 1, 1)
ax0.set_title("Prediction Dynamics")
ax0.set_ylabel("E Amplitude")
h = ax0.imshow(
    EH_pred.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig15.colorbar(h, ax=ax0)
ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("p Amplitude")
h = ax1.imshow(
    pH_pred.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig15.colorbar(h, ax=ax1)
ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$\eta$ Amplitude")
h = ax2.imshow(
    etaH_pred.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig15.colorbar(h, ax=ax2)
plt.subplots_adjust(
    left=0.15, right=1 - 0.01, bottom=0.08, top=1 - 0.08, wspace=None, hspace=0.25
)

fig16 = plt.figure("平面实际演化图", dpi=dpi)
ax0 = plt.subplot(3, 1, 1)

ax0.set_title("Exact Dynamics")
ax0.set_ylabel("E Amplitude")
h = ax0.imshow(
    EExact_h.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig16.colorbar(h, ax=ax0)

ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("p Amplitude")
h = ax1.imshow(
    pExact_h.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig16.colorbar(h, ax=ax1)

ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$\eta$ Amplitude")
h = ax2.imshow(
    etaExact_h.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig16.colorbar(h, ax=ax2)
plt.subplots_adjust(
    left=0.15, right=1 - 0.01, bottom=0.08, top=1 - 0.08, wspace=None, hspace=0.25
)


fig17 = plt.figure("平面误差演化图", dpi=dpi)
ax0 = plt.subplot(3, 1, 1)

ax0.set_title("Error Dynamics")
ax0.set_ylabel("E Error")
h = ax0.imshow(
    EH_pred.T - EExact_h.T,
    interpolation="nearest",
    cmap="PuOr",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig17.colorbar(h, ax=ax0)
ax1 = plt.subplot(3, 1, 2)
ax1.set_ylabel("p Error")
h = ax1.imshow(
    pH_pred.T - pExact_h.T,
    interpolation="nearest",
    cmap="PiYG",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig17.colorbar(h, ax=ax1)
ax2 = plt.subplot(3, 1, 3)
ax2.set_ylabel("$\eta$ Error")
h = ax2.imshow(
    etaH_pred.T - etaExact_h.T,
    interpolation="nearest",
    cmap="seismic",
    extent=[t_lower, t_upper, z_lower, z_upper],
    origin="lower",
    aspect="auto",
)
fig17.colorbar(h, ax=ax2)
plt.subplots_adjust(
    left=0.15, right=1 - 0.01, bottom=0.08, top=1 - 0.08, wspace=None, hspace=0.25
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

scipy.io.savemat(
    "预测结果不要动它_文献75亮MW单孤子.mat",
    {
        "x": x,
        "t": t,
        "EH_pred": EH_pred,
        "pH_pred": pH_pred,
        "etaH_pred": etaH_pred,
        "EExact_h": EExact_h,
        "pExact_h": pExact_h,
        "etaExact_h": etaExact_h,
    },
)
plt.show()
