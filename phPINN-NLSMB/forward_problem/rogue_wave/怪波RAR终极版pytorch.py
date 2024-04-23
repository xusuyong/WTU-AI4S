"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import os
import time

os.environ["DDEBACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh, sqrt, arctan
from scipy import io
import matplotlib

time_string = time.strftime("%Y年%m月%d日%H时%M分%S秒", time.localtime())
folder_name = f"output_{time_string}"
os.makedirs(folder_name, exist_ok=True)
start_time = time.time()
# dde.config.set_default_float("float64")
if dde.backend.backend_name == "paddle":
    import paddle

    sin_tensor = paddle.sin
    exp_tensor = paddle.exp
    cos_tensor = paddle.cos
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
# dde.config.set_default_float("float64")
stride = 5
elevation = 20
azimuth = -40
dpi = 300

z_lower = -3
z_upper = 3
t_lower = -0.5
t_upper = 0.5

# dde.config.set_default_float("float64")
nx = 512
nt = 512
# Creation of the 2D domain (for plotting and input)
x = np.linspace(z_lower, z_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(z_lower, z_upper)
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

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

EExact_u = np.real(EExact)  # (201,256)
EExact_v = np.imag(EExact)
pExact_u = np.real(pExact)
pExact_v = np.imag(pExact)
etaExact_u = np.real(etaExact)


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


def solution(XT):
    X = XT[:, 0:1]
    T = XT[:, 1:2]

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
    EExact_u = real(EExact)  # (201,256)
    EExact_v = imag(EExact)
    pExact_u = real(pExact)
    pExact_v = imag(pExact)
    etaExact_u = real(etaExact)

    return EExact_u, EExact_v, pExact_u, pExact_v, etaExact_u


def output_transform(XT, y):
    Eu = y[:, 0:1]
    Ev = y[:, 1:2]
    pu = y[:, 2:3]
    pv = y[:, 3:4]
    eta = y[:, 4:5]
    X = XT[:, 0:1]
    T = XT[:, 1:2]
    I = 1j
    cos = cos_tensor
    sin = sin_tensor
    cosh = cosh_tensor
    exp = exp_tensor
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
    EExact_u = real(EExact)  # (201,256)
    EExact_v = imag(EExact)
    pExact_u = real(pExact)
    pExact_v = imag(pExact)
    etaExact_u = real(etaExact)
    aaa = (1 - exp(X - z_upper)) * (1 - exp(z_lower - X)) * (
        1 - exp(t_lower - T)
    ) * Eu + EExact_u
    bbb = (1 - exp(X - z_upper)) * (1 - exp(z_lower - X)) * (
        1 - exp(t_lower - T)
    ) * Ev + EExact_v
    ccc = (1 - exp(X - z_upper)) * (1 - exp(z_lower - X)) * (
        1 - exp(t_lower - T)
    ) * pu + pExact_u
    ddd = (1 - exp(X - z_upper)) * (1 - exp(z_lower - X)) * (
        1 - exp(t_lower - T)
    ) * pv + pExact_v
    eee = (1 - exp(X - z_upper)) * (1 - exp(z_lower - X)) * (
        1 - exp(t_lower - T)
    ) * eta + etaExact_u

    return torch.stack([aaa, bbb, ccc, ddd, eee], dim=1)


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

hard_constraint = False
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
    ic_bcs,
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
losshistory, train_state = model.train(
    iterations=iterations,
    display_every=100,
    model_save_path=folder_name + "/",
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
        print("Adding new point:", XTrar[x_ids], "\n")
        data.add_anchors(XTrar[x_ids])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        # model.compile("adam", lr=0.0001)

        losshistory, train_state = model.train(
            iterations=50,
            display_every=100,
            disregard_previous_best=True,
            model_save_path=folder_name + "/",
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
        display_every=100, model_save_path=folder_name + "/", callbacks=[resampler]
    )

elapsed = time.time() - start_time

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

EExact_h = Eh_true.reshape(nt, nx)
pExact_h = ph_true.reshape(nt, nx)
etaExact_h = etah_true.reshape(nt, nx)
EH_pred = Eh_pred.reshape(nt, nx)
pH_pred = ph_pred.reshape(nt, nx)
etaH_pred = etah_pred.reshape(nt, nx)

# Plot predictions
A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -40
dpi = 300

dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=folder_name)


def plot_compare(H_exact, H_pred, tt0, tt1, name):
    fig101 = plt.figure(dpi=dpi)
    ax = plt.subplot(2, 1, 1)

    index = round((tt0 - t_lower) / A * (nt - 1))
    index = max(0, min(index, H_exact.shape[0] - 1))  # 确保 index 在合理范围内
    plt.plot(x, H_exact[index, :], "b-", linewidth=2, label="Exact")
    plt.plot(x, H_pred[index, :], "r--", linewidth=2, label="Prediction")
    ax.set_ylabel(f"$|{name}(t,z)|$")
    ax.set_xlabel("$z$")
    plt.title("t=%s" % tt0)
    plt.legend()
    ax = plt.subplot(2, 1, 2)

    index = round((tt1 - t_lower) / A * (nt - 1))
    index = max(0, min(index, H_exact.shape[0] - 1))  # 确保 index 在合理范围内
    plt.plot(x, H_exact[index, :], "b-", linewidth=2, label="Exact")
    plt.plot(x, H_pred[index, :], "r--", linewidth=2, label="Prediction")
    ax.set_ylabel(f"$|{name}(t,z)|$")
    ax.set_xlabel("$z$")
    plt.title("t=%s" % tt1)
    plt.legend()
    fig101.tight_layout()
    plt.savefig(folder_name + f"/对比图{name}.pdf", dpi="figure")


plot_compare(EExact_h, EH_pred, -2, 2, "E")
plot_compare(pExact_h, pH_pred, -2, 2, "p")
plot_compare(etaExact_h, etaH_pred, -2, 2, "eta")


def plot3d(X, Y, Z, name, cmap):
    fig5 = plt.figure(dpi=dpi, facecolor=None, edgecolor=None, layout="tight")
    ax = fig5.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=stride,  # 指定行的跨度
        cstride=stride,  # 指定列的跨度
        cmap=cmap,  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
        linewidth=0,
        antialiased=False,
    )
    # ax.grid(False)#关闭背景的网格线
    ax.set_xlabel("$z$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$|E(t,z)|$")
    # fig5.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elevation, azimuth)
    plt.savefig(folder_name + f"/3维图{name}.pdf", dpi="figure")


plot3d(X, T, EH_pred, "EH_pred", cmap="Spectral")
plot3d(X, T, pH_pred, "pH_pred", cmap="Spectral")
plot3d(X, T, etaH_pred, "etaH_pred", cmap="Spectral")
plot3d(X, T, EExact_h, "EExact_h", cmap="coolwarm")
plot3d(X, T, pExact_h, "pExact_h", cmap="coolwarm")
plot3d(X, T, etaExact_h, "etaExact_h", cmap="coolwarm")


def plot2d(E, p, eta, name, cmap):
    norm = matplotlib.colors.Normalize(
        vmin=np.min([E, p, eta]),
        vmax=np.max([E, p, eta]),
    )
    fig15 = plt.figure(dpi=dpi, layout="constrained")
    # plt.title(f"{name}")
    ax = fig15.subplots(3, 1, sharex=True)
    ax[0].set_title(f"{name}")
    # ax3.set_title("Prediction Dynamics")
    ax[0].set_ylabel("$|E(t,z)|$", fontsize="medium")
    h = ax[0].imshow(
        E.T,
        interpolation="nearest",
        cmap=cmap,
        extent=[t_lower, t_upper, z_lower, z_upper],
        norm=norm,
        origin="lower",
        aspect="auto",
    )
    ax[1].set_ylabel("$|p(t,z)|$", fontsize="medium")
    h = ax[1].imshow(
        p.T,
        interpolation="nearest",
        cmap=cmap,
        extent=[t_lower, t_upper, z_lower, z_upper],
        norm=norm,
        origin="lower",
        aspect="auto",
    )
    ax[2].set_ylabel("$|\eta(t,z)|$", fontsize="medium")
    h = ax[2].imshow(
        eta.T,
        interpolation="nearest",
        cmap=cmap,
        extent=[t_lower, t_upper, z_lower, z_upper],
        norm=norm,
        origin="lower",
        aspect="auto",
    )
    fig15.colorbar(h, ax=ax)
    # plt.subplots_adjust(
    #     left=0.15, right=1 - 0.01, bottom=0.08, top=1 - 0.08, wspace=None, hspace=0.25
    # )
    plt.savefig(folder_name + f"/投影图{name}.pdf", dpi="figure")


plot2d(EH_pred, pH_pred, etaH_pred, "Prediction", cmap="viridis")
plot2d(EExact_h, pExact_h, etaExact_h, "Exact", cmap="viridis")
plot2d(
    np.abs(EH_pred - EExact_h),
    np.abs(pH_pred - pExact_h),
    np.abs(etaH_pred - etaExact_h),
    "Absolute error",
    cmap="viridis",
)

io.savemat(
    folder_name + f"/预测结果_{os.path.basename(os.getcwd())}.mat",
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
# plt.show()
