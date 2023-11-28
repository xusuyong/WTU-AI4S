"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os

os.environ["DDEBACKEND"] = "tensorflow.compat.v1"
os.makedirs("model", exist_ok=True)
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh, sqrt, arctan

# import torch
import re
import time

start_time = time.time()
# from deepxde.backend import tf


z_lower = -0.5
z_upper = 0.5
t_lower = -2.5
t_upper = 2.5
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
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
I = 1j

alpha_1 = dde.Variable(1.0)  # dtype = float64
alpha_2 = dde.Variable(2.0)
omega_0 = dde.Variable(0.0)
True_alpha_1 = 0.5  # dtype = float64
True_alpha_2 = -1
True_omega_0 = 0.5
true_var_dict = {"True_alpha_1": 0.5, "True_alpha_2": -1, "True_omega_0": 0.5}
# var_dict={"alpha_1":dde.Variable(0.0),"alpha_2":dde.Variable(0.0), "omega_0":dde.Variable(0.0)}

true_var_list = [True_alpha_1, True_alpha_2, True_omega_0]
var_list = [alpha_1, alpha_2, omega_0]
"""X = x[:, 0:1]不能变成X = x[:, 0]!!!"""


def solution(XT):
    from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh

    z = XT[:, 0:1]
    t = XT[:, 1:2]
    EExact = np.exp(((3 * t) / 2 - (65 * z) / 8) * I) * (
        -1
        - 4
        * (-162 * I * z / 17 - 1)
        / (
            (16384 * (-(19 * z) / 128 + (17 * t) / 64) ** 2) / 289
            + 1
            + (26244 * z**2) / 289
        )
    )
    pExact = (
        64
        * I
        / 17
        * (
            -(146088055 * t * z**3) / 33554432
            - (1586899 * t**3 * z) / 8388608
            + (12033042425 * z**4) / 268435456
            + (1419857 * t**4) / 16777216
            + (134257551 * t**2 * z**2) / 33554432
            - (355843677 * z**2) / 134217728
            + (1252815 * t**2) / 33554432
            - (7767453 * z * t) / 33554432
            + 1085773 / 268435456
            - 622796445 * I * z**3 / 33554432
            - 6765201 * I * z * t**2 / 8388608
            + 5595907 * I * z / 33554432
            + 83521 * I * t / 4194304
            + 7561107 * I * z**2 * t / 8388608
        )
        * np.exp(((3 * t) / 2 - (65 * z) / 8) * I)
    ) / (
        4 * ((19 * z) / 128 - (17 * t) / 64) ** 2 + (6561 * z**2) / 1024 + 289 / 4096
    ) ** 2
    etaExact = (
        4624 * t**4
        - 10336 * t**3 * z
        + (218616 * z**2 + 6664) * t**2
        + (-237880 * z**3 + 158440 * z) * t
        + 2449225 * z**4
        - 136934 * z**2
        - 799
    ) / (1565 * z**2 - 76 * z * t + 68 * t**2 + 17) ** 2

    return real(EExact), imag(EExact), real(pExact), imag(pExact), etaExact


def pde(x, y):
    Eu = y[:, 0:1]
    Ev = y[:, 1:2]
    pu = y[:, 2:3]
    pv = y[:, 3:4]
    eta = y[:, 4:5]

    pu_t = dde.grad.jacobian(y, x, i=2, j=1)
    pv_t = dde.grad.jacobian(y, x, i=3, j=1)
    eta_t = dde.grad.jacobian(y, x, i=4, j=1)

    Eu_z = dde.grad.jacobian(y, x, i=0, j=0)
    Ev_z = dde.grad.jacobian(y, x, i=1, j=0)

    Eu_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
    Ev_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)

    f1_u = alpha_1 * Eu_tt - alpha_2 * Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_z
    f1_v = alpha_1 * Ev_tt - alpha_2 * Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_z
    f2_u = 2 * Ev * eta - pv_t + 2 * pu * omega_0
    f2_v = -2 * Eu * eta + pu_t + 2 * pv * omega_0
    f3 = 2 * pv * Ev + 2 * pu * Eu + eta_t

    return [f1_u, f1_v, f2_u, f2_v, f3]


EExact_u, EExact_v, pExact_u, pExact_v, etaExact_u = solution(X_star)

space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域

"""inverse"""
idx = np.random.choice(X_star.shape[0], 5000, replace=False)
X_u_train = X_star[idx, :]

EExact_u_1d, EExact_v_1d, pExact_u_1d, pExact_v_1d, etaExact_u_1d = solution(X_u_train)

observe_y = dde.icbc.PointSetBC(X_u_train, EExact_u_1d, component=0)
observe_y1 = dde.icbc.PointSetBC(X_u_train, EExact_v_1d, component=1)
observe_y2 = dde.icbc.PointSetBC(X_u_train, pExact_u_1d, component=2)
observe_y3 = dde.icbc.PointSetBC(X_u_train, pExact_v_1d, component=3)
observe_y4 = dde.icbc.PointSetBC(X_u_train, etaExact_u_1d, component=4)

data = dde.data.TimePDE(
    geomtime,
    pde,
    # [],
    [observe_y, observe_y1, observe_y2, observe_y3, observe_y4],
    num_domain=20000,
    # num_boundary=20,
    anchors=X_u_train,
    solution=lambda XT: np.hstack((solution(XT))),
    # num_test=10000,
)

layer_size = [2] + [128] * 4 + [5]

net = dde.nn.FNN(layer_size, "sin", "Glorot uniform")


def feature_transform(XT):
    z = XT[:, 0:1]
    t = XT[:, 1:2]
    return concat(
        [(z - z_lower) / (z_upper - z_lower), (t - t_lower) / (t_upper - t_lower)], 1
    )


# net.apply_feature_transform(feature_transform)


def output_transform(XT, y):
    # Uu = y[:, 0]
    # Uv = y[:, 1]
    # Vu = y[:, 2]
    # Vv = y[:, 3]
    # phi0u = y[:, 4]
    # phi0v = y[:, 5]
    z = XT[:, 0]
    t = XT[:, 1]
    # sin = torch.sin
    # tanh = torch.tanh
    # sqrt=torch.sqrt
    # arctan=torch.arctan
    # exp=torch.exp
    # log = torch.log
    # cos = torch.cos
    # sinh = torch.sinh
    # cosh = torch.cosh

    EExact = (
        (-1565 * t**2 + (648 * I + 76 * z) * t - 68 * z**2 + 51)
        * exp(-I / 8 * (-12 * z + 65 * t))
        / (1565 * t**2 - 76 * t * z + 68 * z**2 + 17)
    )

    pExact = (
        (
            9796900 * I * t**4
            + (4056480 - 951520 * I * z) * t**3
            + (-579432 * I + 874464 * I * z**2 - 196992 * z) * t**2
            + (-36448 - 41344 * I * z**3 + 176256 * z**2 - 50592 * I * z) * t
            + 884 * I
            + 18496 * I * z**4
            + 8160 * I * z**2
            - 4352 * z
        )
        * exp(-I / 8 * (-12 * z + 65 * t))
        / (1565 * t**2 - 76 * t * z + 68 * z**2 + 17) ** 2
    )

    etaExact = (
        4624 * z**4
        - 10336 * z**3 * t
        + (218616 * t**2 + 6664) * z**2
        + (-237880 * t**3 + 158440 * t) * z
        + 2449225 * t**4
        - 136934 * t**2
        - 799
    ) / (1565 * t**2 - 76 * t * z + 68 * z**2 + 17) ** 2

    Eu_true = torch.real(EExact)
    Ev_true = torch.imag(EExact)
    pu_true = torch.real(pExact)
    pv_true = torch.imag(pExact)

    # aaa = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * Uu + Eu_true
    # bbb = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * Uv + Ev_true
    # ccc = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * Vu + pu_true
    # ddd = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * Vv + pv_true
    # eee = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * phi0u + phi0u_true
    # fff = (1 - torch.exp(z - z_upper)) * (1 - torch.exp(z_lower - z)) * (1 - torch.exp(t_lower - t)) * phi0v + phi0v_true

    # return torch.stack([aaa, bbb, ccc, ddd, eee, fff], dim=1)
    return torch.stack([Eu_true, Ev_true, pu_true, pv_true, etaExact], dim=1)


# net.apply_output_transform(output_transform)

model = dde.Model(data, net)

iterations = 200
model.compile(
    "adam",
    lr=0.001,
    # loss_weights=[1,1,1,1, 100, 100, 100, 100],
    metrics=["l2 relative error"],
    decay=("inverse time", iterations // 5, 0.5),
    external_trainable_variables=var_list,
)
filenamevar = "variables1.dat"
variable = dde.callbacks.VariableValue(
    var_list, period=period, filename=filenamevar, precision=7  # 加了filename就不会打印了
)
"""Resample"""
resampler = dde.callbacks.PDEPointResampler(
    period=2000, pde_points=True, bc_points=True
)
losshistory, train_state = model.train(
    iterations=iterations,
    display_every=100,
    model_save_path="model/",
    callbacks=[resampler, variable],
)

if 0:
    dde.optimizers.config.set_LBFGS_options(
        # maxcor=50,
        # ftol=1.0 * np.finfo(float).eps,
        # gtol=1e-08,
        maxiter=1000,
        # maxfun=None,
        # maxls=50,
    )
    model.compile(
        "L-BFGS",
        # loss_weights=[1,1,1,1, 100, 100, 100, 100],
        metrics=["l2 relative error"],
        external_trainable_variables=var_list,
    )
    losshistory, train_state = model.train(
        display_every=100, model_save_path="model/", callbacks=[resampler, variable]
    )

# model.restore('model/-205.ckpt')

"""精确解"""
Eh_true = np.sqrt(EExact_u**2 + EExact_v**2).flatten()
ph_true = np.sqrt(pExact_u**2 + pExact_v**2).flatten()
etah_true = np.abs(etaExact_u).flatten()

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

EExact_h = Eh_true.reshape(nt, nx)
pExact_h = ph_true.reshape(nt, nx)
etaExact_h = etah_true.reshape(nt, nx)
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
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line)),
            sep=",",
        )
        for line in lines
    ]
)
l, c = Chat.shape
fig = plt.figure("反问题系数", dpi=dpi, facecolor=None, edgecolor=None)
for i, value in enumerate(true_var_list):
    relative_error = dde.metrics._absolute_percentage_error(value, Chat[-1, i])
    print("relative_error_{}: {}%".format(i, relative_error))
    plt.plot(
        range(0, period * l, period),
        np.ones(Chat[:, 0].shape) * true_var_list[i],
        label="True {}".format(str(true_var_list[i])),
    )
    plt.plot(
        range(0, period * l, period),
        Chat[:, i],
        "--",
        label="{}".format(str(true_var_list[i])),
    )


# plt.plot(range(0, period * l, period), Chat[:, 0], "r-")
# plt.plot(range(0, period * l, period), Chat[:, 1], "k-")
# plt.plot(range(0, period * l, period), Chat[:, 2], "g-")
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 0].shape) * true_var_list[0], "r--")
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 1].shape) * true_var_list[1], "k--")
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 2].shape) * true_var_list[2], "g--")
plt.legend()
plt.xlabel("iterations")
# plt.tight_layout()

# error_lambda_1 = abs((true_var_list[0]-Chat[-1,0])/true_var_list[0]) * 100
# error_lambda_2 = abs((true_var_list[1]-Chat[-1,1])/true_var_list[1]) * 100
# error_omega = abs((true_var_list[2]-Chat[-1,2])/true_var_list[2]) * 100
#
# print('Error l1: %.5f%%' % (error_lambda_1))
# print('Error l2: %.5f%%' % (error_lambda_2))
# print('Error o: %.5f%%' % (error_omega))

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


dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="model/")
# scipy.io.savemat('bound-state.mat',
#                  {'x': x, 't': t, 'elapsed': elapsed, 'X_u_train': X_u_train,
#                   'U_L2_relative_error': U_L2_relative_error, 'V_L2_relative_error': V_L2_relative_error,
#                   'phi0_L2_relative_error': phi0_L2_relative_error,
#                   'UH_pred': UH_pred, 'VH_pred': VH_pred, 'phi0H_pred': phi0H_pred,
#                   'UExact_h': UExact_h, 'VExact_h': VExact_h, 'phi0Exact_h': phi0Exact_h})
plt.show()
