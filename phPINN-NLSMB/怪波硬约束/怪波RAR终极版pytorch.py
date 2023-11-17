import os

os.environ["DDEBACKEND"] = "pytorch"  # pytorch tensorflow.compat.v1
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io  # python读取.mat数据之scipy.io&h5py

# from deepxde.backend import tf
import torch
from numpy import sin, cos, exp, cosh, real, imag
import time

start_time = time.time()


# dde.config.set_default_float("float64")
stride = 5
elevation = 20
azimuth = -40
dpi = 130

z_lower = -3
z_upper = 3
t_lower = -0.5
t_upper = 0.5
N_IC = 100
N_BC = 100
# dde.config.set_default_float("float64")
nx = 512
nt = 512
# Creation of the 2D domain (for plotting and input)
x = np.linspace(z_lower, z_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
I = 1j
EExact = exp(((3 * X) / 2 - (65 * T) / 8) * I) * (
    -1
    - 4
    * (-162 * I * T / 17 - 1)
    / (
        (16384 * (-(19 * T) / 128 + (17 * X) / 64) ** 2) / 289
        + 1
        + (26244 * T**2) / 289
    )
)
pExact = (
    64
    * I
    / 17
    * (
        -(146088055 * X * T**3) / 33554432
        - (1586899 * X**3 * T) / 8388608
        + (12033042425 * T**4) / 268435456
        + (1419857 * X**4) / 16777216
        + (134257551 * X**2 * T**2) / 33554432
        - (355843677 * T**2) / 134217728
        + (1252815 * X**2) / 33554432
        - (7767453 * T * X) / 33554432
        + 1085773 / 268435456
        - 622796445 * I * T**3 / 33554432
        - 6765201 * I * T * X**2 / 8388608
        + 5595907 * I * T / 33554432
        + 83521 * I * X / 4194304
        + 7561107 * I * T**2 * X / 8388608
    )
    * exp(((3 * X) / 2 - (65 * T) / 8) * I)
) / (
    4 * ((19 * T) / 128 - (17 * X) / 64) ** 2 + (6561 * T**2) / 1024 + 289 / 4096
) ** 2
etaExact = (
    4624 * X**4
    - 10336 * X**3 * T
    + (218616 * T**2 + 6664) * X**2
    + (-237880 * T**3 + 158440 * T) * X
    + 2449225 * T**4
    - 136934 * T**2
    - 799
) / (1565 * T**2 - 76 * T * X + 68 * X**2 + 17) ** 2

"""EExact_u = real(EExact) # (201,256)
EExact_v = imag(EExact)
pExact_u = real(pExact)
pExact_v = imag(pExact)
etaExact_u = real(etaExact)  # (201,256)

'''matlab要从底部看，初始（第一行）'''
idx_x = np.random.choice(nx, N_IC, replace=False)  # 输出数组x的行数输出为(0——256-1)，False无放回抽样
x0 = x[idx_x, :]  # 取对应列的数据,(100,1)

Eu0 = EExact_u[0:1, idx_x]  #(1,100)     写成[0, idx_x]会变成(100,)
Ev0 = EExact_v[0:1, idx_x]  # 之所以要写成0:1是因为要确保它是个矩阵
pu0 = pExact_u[0:1, idx_x]
pv0 = pExact_v[0:1, idx_x]
etau0 = etaExact_u[0:1, idx_x]

'''边界'''
idx_t = np.random.choice(nt, N_BC, replace=False)  # 从t中随机抽样50个数据
tb = t[idx_t, :]  # （50，1）
'''lower boundry是第1列（最左边一列），upper boundry是第256列（最右边一列）'''
Eu_lb = EExact_u[idx_t, 0:1] #(100,1)     写成[idx_t, 0]会变成(100,)
Ev_lb = EExact_v[idx_t, 0:1]
pu_lb = pExact_u[idx_t, 0:1]
pv_lb = pExact_v[idx_t, 0:1]
etau_lb = etaExact_u[idx_t, 0:1]

Eu_ub = EExact_u[idx_t, nx-1:nx]
Ev_ub = EExact_v[idx_t, nx-1:nx]
pu_ub = pExact_u[idx_t, nx-1:nx]
pv_ub = pExact_v[idx_t, nx-1:nx]
etau_ub = etaExact_u[idx_t, nx-1:nx]

X0 = np.concatenate((x0, 0 * x0 + t_lower), axis=1)  # (x0, 0)，axis=0竖直拼接，1水平拼接，（50，2）
X_lb = np.concatenate((0 * tb + z_lower, tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + z_upper, tb), 1)  # (ub[0], tb)
X_u_train = np.vstack([X_lb, X0, X_ub])  #(300,2)

Eu_icbc = np.vstack([Eu_lb, Eu0.T, Eu_ub])  # (300,1)
Ev_icbc = np.vstack([Ev_lb, Ev0.T, Ev_ub])
pu_icbc = np.vstack([pu_lb, pu0.T, pu_ub])
pv_icbc = np.vstack([pv_lb, pv0.T, pv_ub])
etau_icbc = np.vstack([etau_lb, etau0.T, etau_ub])


# etav_icbc = np.vstack([etav_lb, etav0.T, etav_ub])
# print(X_u_train,'\n', etau_icbc)
# exit()"""

# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域


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
    Eu_t = dde.grad.jacobian(y, x, i=0, j=1)  # 一阶导用jacobian，二阶导用hessian
    Ev_t = dde.grad.jacobian(y, x, i=1, j=1)
    # pu_t = dde.grad.jacobian(y, x, i=2, j=1)  # 一阶导用jacobian，二阶导用hessian
    # pv_t = dde.grad.jacobian(y, x, i=3, j=1)
    # eta_t = dde.grad.jacobian(y, x, i=4, j=1)
    #
    # Eu_z = dde.grad.jacobian(y, x, i=0, j=0)  # 一阶导用jacobian，二阶导用hessian
    # Ev_z = dde.grad.jacobian(y, x, i=1, j=0)
    pu_z = dde.grad.jacobian(y, x, i=2, j=0)  # 一阶导用jacobian，二阶导用hessian
    pv_z = dde.grad.jacobian(y, x, i=3, j=0)
    eta_z = dde.grad.jacobian(y, x, i=4, j=0)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    # Eu_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
    # Ev_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)
    Eu_zz = dde.grad.hessian(y, x, component=0, i=0, j=0)
    Ev_zz = dde.grad.hessian(y, x, component=1, i=0, j=0)

    f1_u = 0.5 * Eu_zz + Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_t
    f1_v = 0.5 * Ev_zz + Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_t
    f2_u = 2 * Ev * eta - pv_z + pu
    f2_v = -2 * Eu * eta + pu_z + pv
    f3 = 2 * pv * Ev + 2 * pu * Eu + eta_z

    return [f1_u, f1_v, f2_u, f2_v, f3]


def output_transform(x, y):
    Eu = y[:, 0]
    Ev = y[:, 1]
    pu = y[:, 2]
    pv = y[:, 3]
    eta = y[:, 4]
    X = x[:, 0]
    T = x[:, 1]
    I = 1j
    E_true = torch.exp(((3 * X) / 2 - (65 * T) / 8) * I) * (
        -1
        - 4
        * (-162 * I * T / 17 - 1)
        / (
            (16384 * (-(19 * T) / 128 + (17 * X) / 64) ** 2) / 289
            + 1
            + (26244 * T**2) / 289
        )
    )
    p_true = (
        64
        * I
        / 17
        * (
            -(146088055 * X * T**3) / 33554432
            - (1586899 * X**3 * T) / 8388608
            + (12033042425 * T**4) / 268435456
            + (1419857 * X**4) / 16777216
            + (134257551 * X**2 * T**2) / 33554432
            - (355843677 * T**2) / 134217728
            + (1252815 * X**2) / 33554432
            - (7767453 * T * X) / 33554432
            + 1085773 / 268435456
            - 622796445 * I * T**3 / 33554432
            - 6765201 * I * T * X**2 / 8388608
            + 5595907 * I * T / 33554432
            + 83521 * I * X / 4194304
            + 7561107 * I * T**2 * X / 8388608
        )
        * torch.exp(((3 * X) / 2 - (65 * T) / 8) * I)
    ) / (
        4 * ((19 * T) / 128 - (17 * X) / 64) ** 2 + (6561 * T**2) / 1024 + 289 / 4096
    ) ** 2
    eta_true = (
        4624 * X**4
        - 10336 * X**3 * T
        + (218616 * T**2 + 6664) * X**2
        + (-237880 * T**3 + 158440 * T) * X
        + 2449225 * T**4
        - 136934 * T**2
        - 799
    ) / (1565 * T**2 - 76 * T * X + 68 * X**2 + 17) ** 2
    Eu_true = real(E_true)
    Ev_true = imag(E_true)
    pu_true = real(p_true)
    pv_true = imag(p_true)
    aaa = (1 - torch.exp(X - z_upper)) * (1 - torch.exp(z_lower - X)) * (
        1 - torch.exp(t_lower - T)
    ) * Eu + Eu_true
    bbb = (1 - torch.exp(X - z_upper)) * (1 - torch.exp(z_lower - X)) * (
        1 - torch.exp(t_lower - T)
    ) * Ev + Ev_true
    ccc = (1 - torch.exp(X - z_upper)) * (1 - torch.exp(z_lower - X)) * (
        1 - torch.exp(t_lower - T)
    ) * pu + pu_true
    ddd = (1 - torch.exp(X - z_upper)) * (1 - torch.exp(z_lower - X)) * (
        1 - torch.exp(t_lower - T)
    ) * pv + pv_true
    eee = (1 - torch.exp(X - z_upper)) * (1 - torch.exp(z_lower - X)) * (
        1 - torch.exp(t_lower - T)
    ) * eta + eta_true

    return torch.stack([aaa, bbb, ccc, ddd, eee], dim=1)


# observe_y = dde.icbc.PointSetBC(X_u_train, Eu_icbc, component=0)
# observe_y1 = dde.icbc.PointSetBC(X_u_train, Ev_icbc, component=1)
# observe_y2 = dde.icbc.PointSetBC(X_u_train, pu_icbc, component=2)
# observe_y3 = dde.icbc.PointSetBC(X_u_train, pv_icbc, component=3)
# observe_y4 = dde.icbc.PointSetBC(X_u_train, etau_icbc, component=4)
"""初始条件就是t等于-5的时候"""
# 初始条件满足init_cond_u或init_cond_v函数
data = dde.data.TimePDE(
    geomtime,
    pde,
    [],
    # [ observe_y, observe_y1, observe_y2, observe_y3, observe_y4],
    num_domain=25000,
    train_distribution="pseudo",
)  # 在内部取10000个点，在边界取20个点，在初始取200个点,"pseudo" (pseudorandom)伪随机分布
# 前面输出pde的loss，后面输出初始、边界的loss
# Network architecture
net = dde.nn.FNN([2] + [128] * 6 + [5], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)


model = dde.Model(data, net)

resampler = dde.callbacks.PDEPointResampler(period=5000)

model.compile(
    "adam",
    lr=0.001,
    loss="MSE",
    # decay=("inverse time", 5000, 0.5),
    decay=("step", 5000, 0.7),
    # loss_weights=[1, 1, 1, 1, 1, 100, 100, 100, 100, 100],
    # loss_weights=[0.01, 0.01, 0.01, 0.01, 0.01],
)
losshistory, train_state = model.train(
    iterations=100, display_every=100, callbacks=[resampler]
)

# dde.optimizers.config.set_LBFGS_options(
#     maxcor=50,
#     ftol=1.0 * np.finfo(float).eps,
#     gtol=1e-08,
#     maxiter=50000,
#     maxfun=50000,
#     maxls=50,
# )
# model.compile("L-BFGS",
#               # loss_weights=[1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
#               )
# losshistory, train_state = model.train(display_every=100,
#                                        # callbacks=[resampler]
#                                        )

# XT = geomtime.random_points(100000)
# err = 1
# while err > 0.005:
#     f = model.predict(XT, operator=pde)
#     err_eq = np.absolute(f)
#     err = np.mean(err_eq)
#     print("Mean residual: %.3e" % (err))
#
#     x_id = np.argmax(err_eq)
#     print("Adding new point:", XT[x_id], "\n")
#     data.add_anchors(XT[x_id])
#     early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
#     model.compile("adam", lr=1e-3)
#     model.train(epochs=10000, disregard_previous_best=True, callbacks=[early_stopping])
#     model.compile("L-BFGS")
#     losshistory, train_state = model.train(display_every=100)


"""RAR"""
for i in range(5):  # 一下添加几个点，总共这些次
    XT = geomtime.random_points(100000)
    f0 = model.predict(XT, operator=pde)
    f1 = np.sum(f0, axis=0).flatten()
    err_eq = np.absolute(f1)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    # err_eq = torch.tensor(err_eq)
    # x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()#返回最大的前十个
    x_ids = np.argsort(err_eq)[-200:]
    # print(x_ids)

    for elem in x_ids:
        print("Adding new point:", XT[elem], "\n")
        data.add_anchors(XT[elem])
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    # model.compile("adam", lr=0.0007)
    # losshistory, train_state = model.train(
    #     iterations=10000, disregard_previous_best=True, callbacks=[early_stopping]
    # )

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)

"""精确解"""
EExact_h = np.abs(EExact)  # （201，256）
Eh_true = EExact_h.flatten()  # (51456,)
pExact_h = np.abs(pExact)
ph_true = pExact_h.flatten()
etaExact_h = np.abs(etaExact)
etah_true = etaExact_h.flatten()
# Make prediction
"""预测解"""
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

A = t_upper - t_lower

fig101 = plt.figure("E对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
tt0 = 0
index = round(
    (tt0 - t_lower) / A * (nt - 1)
)  # index只能是0-200（总共有201行,=0时索引第1个数,=200时索引第201）
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
tt1 = 0.1
index = round((tt1 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
plt.legend()
fig101.tight_layout()

fig102 = plt.figure("p对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
# tt0 = -5
index = round((tt0 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
# tt1 = 5
index = round((tt1 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
plt.legend()
fig102.tight_layout()

fig103 = plt.figure("eta对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
# tt0 = -5
index = round((tt0 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
# tt1 = 5
index = round((tt1 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
plt.legend()
fig103.tight_layout()


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

# fig8 = plt.figure("3d真解E", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig8.add_subplot(projection='3d')
# surf=ax.plot_surface(X, T, EExact_h,
#                        rstride=stride,  # 指定行的跨度
#                        cstride=stride,  # 指定列的跨度
#                        cmap='coolwarm',  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
#                        linewidth=0,
#                        antialiased=False)  # 抗锯齿
# # ax.grid(False)#关闭背景的网格线
# ax.set(xlabel='$z$', ylabel='$t$', zlabel='$|E(t,z)|$')
# # fig8.colorbar(surf, shrink=0.5, aspect=5)
# ax.view_init(elevation, azimuth)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠
#
# fig9 = plt.figure("3d真解p", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig9.add_subplot(projection='3d')
# surf=ax.plot_surface(X, T, pExact_h,
#                        rstride=stride,  # 指定行的跨度
#                        cstride=stride,  # 指定列的跨度
#                        cmap='coolwarm',  # 设置颜色映射
#                        linewidth=0,
#                        antialiased=False)  # 抗锯齿
# # ax.grid(False)#关闭背景的网格线
# ax.set(xlabel='$z$', ylabel='$t$', zlabel='$|p(t,z)|$')
# # fig9.colorbar(surf, shrink=0.5, aspect=5)
# ax.view_init(elevation, azimuth)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠
#
#
# fig10 = plt.figure("3d真解eta", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig10.add_subplot(projection='3d')
# surf=ax.plot_surface(X, T, etaExact_h,
#                        rstride=stride,  # 指定行的跨度
#                        cstride=stride,  # 指定列的跨度
#                        cmap='coolwarm',  # 设置颜色映射
#                        linewidth=0,
#                        antialiased=False)  # 抗锯齿
# # ax.grid(False)#关闭背景的网格线
# ax.set(xlabel='$z$', ylabel='$t$', zlabel='$|\eta(t,z)|$')
# # fig10.colorbar(surf, shrink=0.5, aspect=5)
# ax.view_init(elevation, azimuth)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠

# dde薛定谔里的图
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
# ax0.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax0.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax0.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax1.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
ax1.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax2.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
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
# ax0.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax0.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax0.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax1.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
ax1.plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax1.plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax2.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,clip_on=False)
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
    cmap="ocean_r",  # PuOr
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
    cmap="ocean_r",
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
    cmap="ocean_r",  # seismic
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax2)
fig17.colorbar(h0, ax=[ax0, ax1, ax2], location="right")
# plt.subplots_adjust(left=0.15, right=1-0.01,bottom=0.08, top=1-0.08,wspace=None, hspace=0.25)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠

# plt.savefig(r'F:\QQ截屏录屏\亮亮暗2\d')

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

scipy.io.savemat(
    "预测结果_怪波.mat",
    {
        "x": x,
        "t": t,
        "nx": nx,
        "nt": nt,
        "elapsed": elapsed,
        "X_u_train": X_u_train,
        "E_L2_relative_error": E_L2_relative_error,
        "p_L2_relative_error": p_L2_relative_error,
        "eta_L2_relative_error": eta_L2_relative_error,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "EH_pred": EH_pred,
        "pH_pred": pH_pred,
        "etaH_pred": etaH_pred,
        "EExact_h": EExact_h,
        "pExact_h": pExact_h,
        "etaExact_h": etaExact_h,
    },
)

plt.show()
