"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os

os.environ["DDEBACKEND"] = "pytorch"
os.makedirs("output_dir", exist_ok=True)
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh, sqrt, arctan
from scipy import io
import time

start_time = time.time()

if dde.backend.backend_name == "paddle":
    import paddle

    sin_tensor = paddle.sin
    cos_tensor = paddle.cos
    exp_tensor = paddle.exp
    cosh_tensor = paddle.cosh
    concat = paddle.concat
elif dde.backend.backend_name == "pytorch":
    import torch

    sin_tensor = torch.sin
    cos_tensor = torch.cos
    exp_tensor = torch.exp
    cosh_tensor = torch.cosh
    concat = torch.cat
else:
    from deepxde.backend import tf

    sin_tensor = tf.math.sin
    cos_tensor = tf.math.cos
    exp_tensor = tf.math.exp
    cosh_tensor = tf.math.cosh
    concat = tf.concat
z_lower = -2
z_upper = 2
t_lower = -3
t_upper = 3
nx = 512
nt = 512
# Creation of the 2D domain (for plotting and input)
x = np.linspace(z_lower, z_upper, nx)
t = np.linspace(t_lower, t_upper, nt)
X, T = np.meshgrid(x, t)
# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))


# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)  # 结合一下，变成时空区域


# The "physics-informed" part of the loss
def pde(x, y):  # 这里x其实是x和t，y其实是u和v
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

    omega = -1

    f1_u = 0.5 * Eu_tt + Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_z
    f1_v = 0.5 * Ev_tt + Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_z
    f2_u = 2 * Ev * eta - pv_t + 2 * pu * omega
    f2_v = -2 * Eu * eta + pu_t + 2 * pv * omega
    f3 = 2 * pv * Ev + 2 * pu * Eu + eta_t

    return [f1_u, f1_v, f2_u, f2_v, f3]


# Boundary and Initial conditions
def solution(XT):
    x = XT[:, 0:1]
    t = XT[:, 1:2]

    Eu_true = 2 * cos(2 * t) / cosh(2 * t + 6 * x)

    Ev_true = -2 * sin(2 * t) / cosh(2 * t + 6 * x)

    pu_true = (
        (exp(-2 * t - 6 * x) - exp(2 * t + 6 * x))
        * cos(2 * t)
        / cosh(2 * t + 6 * x) ** 2
    )
    pv_true = (
        -(exp(-2 * t - 6 * x) - exp(2 * t + 6 * x))
        * sin(2 * t)
        / cosh(2 * t + 6 * x) ** 2
    )
    eta_true = (cosh(2 * t + 6 * x) ** 2 - 2) / cosh(2 * t + 6 * x) ** 2

    return Eu_true, Ev_true, pu_true, pv_true, eta_true


Eu_true, Ev_true, pu_true, pv_true, eta_true = solution(X_star)


def output_transform(XT, y):
    Eu = y[:, 0:1]
    Ev = y[:, 1:2]
    pu = y[:, 2:3]
    pv = y[:, 3:4]
    eta = y[:, 4:5]
    x = XT[:, 0:1]
    t = XT[:, 1:2]
    I = 1j
    cos = cos_tensor
    sin = sin_tensor
    cosh = cosh_tensor
    exp = exp_tensor

    Eu_true = 2 * cos(2 * t) / cosh(2 * t + 6 * x)

    Ev_true = -2 * sin(2 * t) / cosh(2 * t + 6 * x)

    pu_true = (
        (exp(-2 * t - 6 * x) - exp(2 * t + 6 * x))
        * cos(2 * t)
        / cosh(2 * t + 6 * x) ** 2
    )
    pv_true = (
        -(exp(-2 * t - 6 * x) - exp(2 * t + 6 * x))
        * sin(2 * t)
        / cosh(2 * t + 6 * x) ** 2
    )
    eta_true = (cosh(2 * t + 6 * x) ** 2 - 2) / cosh(2 * t + 6 * x) ** 2

    aaa = (1 - exp(x - z_upper)) * (1 - exp(z_lower - x)) * (
        1 - exp(t_lower - t)
    ) * Eu + Eu_true
    bbb = (1 - exp(x - z_upper)) * (1 - exp(z_lower - x)) * (
        1 - exp(t_lower - t)
    ) * Ev + Ev_true
    ccc = (1 - exp(x - z_upper)) * (1 - exp(z_lower - x)) * (
        1 - exp(t_lower - t)
    ) * pu + pu_true
    ddd = (1 - exp(x - z_upper)) * (1 - exp(z_lower - x)) * (
        1 - exp(t_lower - t)
    ) * pv + pv_true
    eee = (1 - exp(x - z_upper)) * (1 - exp(z_lower - x)) * (
        1 - exp(t_lower - t)
    ) * eta + eta_true

    return concat([aaa, bbb, ccc, ddd, eee], 1)


"""forward"""
ic = X_star[:, 1] == t_lower
idx_ic = np.random.choice(np.where(ic)[0], 200, replace=False)
lb = X_star[:, 0] == z_lower
idx_lb = np.random.choice(np.where(lb)[0], 200, replace=False)
ub = X_star[:, 0] == z_upper
idx_ub = np.random.choice(np.where(ub)[0], 200, replace=False)
icbc_idx = np.hstack((idx_lb, idx_ic, idx_ub))
X_u_train = X_star[icbc_idx]

"""inverse"""
# idx = np.random.choice(X_star.shape[0], 10000, replace=False)
# X_u_train = X_star[idx, :]  # (2000,2)

Eu_train, Ev_train, pu_train, pv_train, eta_train = solution(X_u_train)

observe_y = dde.icbc.PointSetBC(X_u_train, Eu_train, component=0)
observe_y1 = dde.icbc.PointSetBC(X_u_train, Ev_train, component=1)
observe_y2 = dde.icbc.PointSetBC(X_u_train, pu_train, component=2)
observe_y3 = dde.icbc.PointSetBC(X_u_train, pv_train, component=3)
observe_y4 = dde.icbc.PointSetBC(X_u_train, eta_train, component=4)
# Network architecture
PFNN = True
net = (
    dde.nn.PFNN(
        [
            2,
            [16, 16, 16, 16, 16],
            [16, 16, 16, 16, 16],
            [16, 16, 16, 16, 16],
            [16, 16, 16, 16, 16],
            [16, 16, 16, 16, 16],
            [16, 16, 16, 16, 16],
            5,
        ],
        "tanh",
        "Glorot normal",
    )
    if PFNN
    else dde.nn.FNN([2] + [64] * 6 + [5], "tanh", "Glorot normal")
)

hard_constraint = True
if hard_constraint:
    net.apply_output_transform(output_transform)
ic_bcs = (
    []
    if hard_constraint
    else [observe_y, observe_y1, observe_y2, observe_y3, observe_y4]
)
data = dde.data.TimePDE(
    geomtime,
    pde,
    ic_bcs=ic_bcs,
    num_domain=20000,
    solution=lambda XT: np.hstack((solution(XT))),
)

model = dde.Model(data, net)

resampler = dde.callbacks.PDEPointResampler(period=5000)
loss_weights = [1, 1, 1, 1, 1] + np.full(len(ic_bcs), 100).tolist()
iterations = 3
model.compile(
    "adam",
    lr=0.001,
    loss="MSE",
    metrics=["l2 relative error"],
    decay=("inverse time", iterations // 3, 0.5),
    loss_weights=loss_weights,
)

# model.restore("output_dir/-350.pt")

losshistory, train_state = model.train(
    iterations=100,
    display_every=100,
    model_save_path="output_dir/",
    callbacks=[resampler],
)

RAR = True
if RAR:
    for i in range(5):  # 一下添加几个点，总共这些次
        XTrar = geomtime.random_points(100000)
        f = model.predict(XTrar, operator=pde)
        err_eq = np.absolute(np.array(f))
        err_eq = np.sum(err_eq, axis=0).flatten()
        err = np.mean(err_eq)
        print("Mean residual: %.3e" % (err))
        x_ids = np.argsort(err_eq)[-100:]

        # for elem in x_ids:
        print("Adding new point:", XTrar[x_ids], "\n")
        data.add_anchors(XTrar[x_ids])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        # model.compile("adam", lr=0.0001)

        losshistory, train_state = model.train(
            iterations=50,
            display_every=100,
            disregard_previous_best=True,
            model_save_path="output_dir/",
            callbacks=[early_stopping, resampler],
        )

LBFGS = True
if LBFGS:
    dde.optimizers.config.set_LBFGS_options(
        maxcor=50,
        ftol=1.0 * np.finfo(float).eps,
        gtol=1e-08,
        maxiter=100,
        maxfun=None,
        maxls=50,
    )
    model.compile(
        "L-BFGS",
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
    )
    losshistory, train_state = model.train(
        display_every=100, model_save_path="output_dir/", callbacks=[resampler]
    )

elapsed = time.time() - start_time

# 精确解
Eh_true = np.sqrt(Eu_true**2 + Ev_true**2).flatten()
ph_true = np.sqrt(pu_true**2 + pv_true**2).flatten()
etah_true = np.abs(eta_true).flatten()
# Make prediction
# 预测解
prediction = model.predict(X_star)
Eu_pred = prediction[:, 0]
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

"""精确解"""
EExact_h = Eh_true.reshape(nt, nx)
pExact_h = ph_true.reshape(nt, nx)
etaExact_h = etah_true.reshape(nt, nx)
"""预测解"""
EH_pred = Eh_pred.reshape(nt, nx)
pH_pred = ph_pred.reshape(nt, nx)
etaH_pred = etah_pred.reshape(nt, nx)

# Plot predictions
A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -40
dpi = 300

fig101 = plt.figure("E对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
tt0 = -2
index = round((tt0 - t_lower) / A * (nt - 1))
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
tt1 = 2
index = round((tt1 - t_lower) / A * (nt - 1))
plt.plot(x, EExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, EH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|E(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
plt.legend()
fig101.tight_layout()

fig102 = plt.figure("p对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
# tt0 = -2
index = round((tt0 - t_lower) / A * (nt - 1))
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
# tt1 = 2
index = round((tt1 - t_lower) / A * (nt - 1))
plt.plot(x, pExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, pH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|p(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
plt.legend()
fig102.tight_layout()

fig103 = plt.figure("eta对比图", dpi=dpi)
ax = plt.subplot(2, 1, 1)
# tt0 = -2
index = round((tt0 - t_lower) / A * (nt - 1))
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt0)
plt.legend()
ax = plt.subplot(2, 1, 2)
# tt1 = 2
index = round((tt1 - t_lower) / A * (nt - 1))
plt.plot(x, etaExact_h[index, :], "b-", linewidth=2, label="Exact")
plt.plot(x, etaH_pred[index, :], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$|\eta(t,z)|$")
ax.set_xlabel("$z$")
plt.title("t=%s" % tt1)
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

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir="output_dir/"
)

io.savemat(
    "output_dir/文献75亮MW单孤子.mat",
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
    },
)
plt.show()
