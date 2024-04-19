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

    sin_tesnor = paddle.sin
    exp_tensor = paddle.exp
    cos_tesnor = paddle.cos
    cosh_tensor = paddle.cosh
    concat = paddle.concat
elif dde.backend.backend_name == "pytorch":
    import torch

    sin_tesnor = torch.sin
    cos_tesnor = torch.cos
    exp_tensor = torch.exp
    cosh_tensor = torch.cosh
    concat = torch.cat
else:
    from deepxde.backend import tf

    sin_tesnor = tf.math.sin
    cos_tesnor = tf.math.cos
    exp_tensor = tf.math.exp
    cosh_tensor = tf.math.cosh
    concat = tf.concat

z_lower = -5
z_upper = 5
t_lower = -5
t_upper = 5

# dde.config.set_default_float("float64")
nx = 512
nt = 512
# Creation of the 2D domain (for plotting and input)
x = np.linspace(z_lower, z_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(z_lower, z_upper)  # 先定义空间
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)  # 再定义时间
geomtime = dde.geometry.GeometryXTime(
    space_domain, time_domain
)  # 结合一下，变成时空区域
I = 1j
EExact = (
    -(12 + 16 * I) * np.exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
    + (-12 + 16 * I) * np.exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
    + 400 * np.exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
    + 400 * np.exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
) / (
    (48 + 64 * I) * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
    + (48 - 64 * I) * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
    + 100 * np.exp(2 * T - (58 * X) / 13 + 2)
    + 100 * np.exp(2 * T + (26 * X) / 29 + 2)
    + np.exp(4 * T - (1344 * X) / 377 + 4)
    + 400
)
pExact = (
    16
    * (
        42688 * I * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + 832 * I * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        + 352 * I * np.exp(4 * T - (1344 * X) / 377 + 4)
        + 26000 * I * np.exp(2 * T + (26 * X) / 29 + 2)
        + 34800 * I * np.exp(2 * T - (58 * X) / 13 + 2)
        - 14384 * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + 20176 * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        - 135 * np.exp(4 * T - (1344 * X) / 377 + 4)
        + 27300 * np.exp(2 * T + (26 * X) / 29 + 2)
        + 14500 * np.exp(2 * T - (58 * X) / 13 + 2)
        + 150800
    )
    * (
        283 * I * np.exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
        - 339 * I * np.exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
        - 8700 * I * np.exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
        - 6500 * I * np.exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
        + 206 * np.exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
        + 398 * np.exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
        - 5800 * np.exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
        - 2600 * np.exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
    )
    / (
        142129
        * (
            64 * I * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            - 64 * I * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * np.exp(2 * T - (58 * X) / 13 + 2)
            + 48 * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 48 * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * np.exp(2 * T + (26 * X) / 29 + 2)
            + np.exp(4 * T - (1344 * X) / 377 + 4)
            + 400
        )
        ** 2
    )
)
etaExact = (
    (675584 + 2316288 * I)
    * np.exp((4 + 2 * I) * T + 4 - (1344 / 377 + 352 * I / 377) * X)
    - (3222400 + 6003200 * I)
    * np.exp((4 - I) * T + 4 + (-334 / 377 + 176 * I / 377) * X)
    + (9843200 + 14182400 * I)
    * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
    - (4860800 + 4774400 * I)
    * np.exp((4 - I) * T + 4 + (-2354 / 377 + 176 * I / 377) * X)
    - (40928 + 13696 * I) * np.exp((6 + I) * T + 6 - (2016 / 377 + 176 * I / 377) * X)
    + (-4860800 + 4774400 * I)
    * np.exp((4 + I) * T + 4 - (2354 / 377 + 176 * I / 377) * X)
    + (-3222400 + 6003200 * I)
    * np.exp((4 + I) * T + 4 - (334 / 377 + 176 * I / 377) * X)
    + (675584 - 2316288 * I)
    * np.exp((4 - 2 * I) * T + 4 + (-1344 / 377 + 352 * I / 377) * X)
    + (9843200 - 14182400 * I)
    * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
    + (-40928 + 13696 * I) * np.exp((6 - I) * T + 6 + (-2016 / 377 + 176 * I / 377) * X)
    - 377 * np.exp(8 * T - (2688 * X) / 377 + 8)
    + 6960000 * np.exp(2 * T - (58 * X) / 13 + 2)
    - 13520000 * np.exp(2 * T + (26 * X) / 29 + 2)
    - 3770000 * np.exp(4 * T + (52 * X) / 29 + 4)
    - 3770000 * np.exp(4 * T - (116 * X) / 13 + 4)
    - 13844800 * np.exp(4 * T - (1344 * X) / 377 + 4)
    - 33800 * np.exp(6 * T - (3026 * X) / 377 + 6)
    + 17400 * np.exp(6 * T - (1006 * X) / 377 + 6)
    - 60320000
) / (
    377
    * (
        64 * I * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        - 64 * I * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        + 100 * np.exp(2 * T - (58 * X) / 13 + 2)
        + 48 * np.exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + 48 * np.exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        + 100 * np.exp(2 * T + (26 * X) / 29 + 2)
        + np.exp(4 * T - (1344 * X) / 377 + 4)
        + 400
    )
    ** 2
)

# EExact = data1['q1']  # (201,256)
EExact_u = np.real(EExact)  # (201,256)
EExact_v = np.imag(EExact)
# EExact_h = np.sqrt(EExact_u ** 2 + EExact_v ** 2)

# pExact = data1['q2']  # (201,256)
pExact_u = np.real(pExact)
pExact_v = np.imag(pExact)
# pExact_h = np.sqrt(pExact_u ** 2 + pExact_v ** 2)

# etaExact = data1['q3']  # (201,256)
etaExact_u = np.real(etaExact)  # (201,256)
etaExact_v = np.imag(etaExact)


# etaExact_h = np.sqrt(etaExact_u ** 2 + etaExact_v ** 2)
def pde(x, y):  # 这里x其实是x和t，y其实是u和v
    Eu = y[:, 0:1]
    Ev = y[:, 1:2]
    pu = y[:, 2:3]
    pv = y[:, 3:4]
    etau = y[:, 4:5]

    # In 'jacobian', i is the output component and j is the input component
    pu_t = dde.grad.jacobian(y, x, i=2, j=1)  # 一阶导用jacobian，二阶导用hessian
    pv_t = dde.grad.jacobian(y, x, i=3, j=1)
    etau_t = dde.grad.jacobian(y, x, i=4, j=1)

    Eu_z = dde.grad.jacobian(y, x, i=0, j=0)  # 一阶导用jacobian，二阶导用hessian
    Ev_z = dde.grad.jacobian(y, x, i=1, j=0)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    Eu_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
    Ev_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)

    f1_u = Eu_tt + 2 * Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_z
    f1_v = Ev_tt + 2 * Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_z
    f2_u = 2 * Ev * etau - pv_t + 2 * pu
    f2_v = -2 * Eu * etau + pu_t + 2 * pv
    f3_u = 2 * pv * Ev + 2 * pu * Eu + etau_t

    return [f1_u, f1_v, f2_u, f2_v, f3_u]


def solution(XT):
    X = XT[:, 0:1]
    T = XT[:, 1:2]

    EExact = (
        -(12 + 16 * I) * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
        + (-12 + 16 * I) * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
        + 400 * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
        + 400 * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
    ) / (
        (48 + 64 * I) * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + (48 - 64 * I) * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        + 100 * exp(2 * T - (58 * X) / 13 + 2)
        + 100 * exp(2 * T + (26 * X) / 29 + 2)
        + exp(4 * T - (1344 * X) / 377 + 4)
        + 400
    )

    pExact = (
        16
        * (
            42688 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 832 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 352 * I * exp(4 * T - (1344 * X) / 377 + 4)
            + 26000 * I * exp(2 * T + (26 * X) / 29 + 2)
            + 34800 * I * exp(2 * T - (58 * X) / 13 + 2)
            - 14384 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 20176 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            - 135 * exp(4 * T - (1344 * X) / 377 + 4)
            + 27300 * exp(2 * T + (26 * X) / 29 + 2)
            + 14500 * exp(2 * T - (58 * X) / 13 + 2)
            + 150800
        )
        * (
            283 * I * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
            - 339 * I * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
            - 8700 * I * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
            - 6500 * I * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
            + 206 * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
            + 398 * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
            - 5800 * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
            - 2600 * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
        )
        / (
            142129
            * (
                64 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
                - 64 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
                + 100 * exp(2 * T - (58 * X) / 13 + 2)
                + 48 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
                + 48 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
                + 100 * exp(2 * T + (26 * X) / 29 + 2)
                + exp(4 * T - (1344 * X) / 377 + 4)
                + 400
            )
            ** 2
        )
    )
    etaExact = (
        (675584 + 2316288 * I)
        * exp((4 + 2 * I) * T + 4 - (1344 / 377 + 352 * I / 377) * X)
        - (3222400 + 6003200 * I)
        * exp((4 - I) * T + 4 + (-334 / 377 + 176 * I / 377) * X)
        + (9843200 + 14182400 * I)
        * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        - (4860800 + 4774400 * I)
        * exp((4 - I) * T + 4 + (-2354 / 377 + 176 * I / 377) * X)
        - (40928 + 13696 * I) * exp((6 + I) * T + 6 - (2016 / 377 + 176 * I / 377) * X)
        + (-4860800 + 4774400 * I)
        * exp((4 + I) * T + 4 - (2354 / 377 + 176 * I / 377) * X)
        + (-3222400 + 6003200 * I)
        * exp((4 + I) * T + 4 - (334 / 377 + 176 * I / 377) * X)
        + (675584 - 2316288 * I)
        * exp((4 - 2 * I) * T + 4 + (-1344 / 377 + 352 * I / 377) * X)
        + (9843200 - 14182400 * I)
        * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + (-40928 + 13696 * I)
        * exp((6 - I) * T + 6 + (-2016 / 377 + 176 * I / 377) * X)
        - 377 * exp(8 * T - (2688 * X) / 377 + 8)
        + 6960000 * exp(2 * T - (58 * X) / 13 + 2)
        - 13520000 * exp(2 * T + (26 * X) / 29 + 2)
        - 3770000 * exp(4 * T + (52 * X) / 29 + 4)
        - 3770000 * exp(4 * T - (116 * X) / 13 + 4)
        - 13844800 * exp(4 * T - (1344 * X) / 377 + 4)
        - 33800 * exp(6 * T - (3026 * X) / 377 + 6)
        + 17400 * exp(6 * T - (1006 * X) / 377 + 6)
        - 60320000
    ) / (
        377
        * (
            64 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            - 64 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * exp(2 * T - (58 * X) / 13 + 2)
            + 48 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 48 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * exp(2 * T + (26 * X) / 29 + 2)
            + exp(4 * T - (1344 * X) / 377 + 4)
            + 400
        )
        ** 2
    )
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
    cos = cos_tesnor
    sin = sin_tesnor
    cosh = cosh_tensor
    exp = exp_tensor
    EExact = (
        -(12 + 16 * I) * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
        + (-12 + 16 * I) * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
        + 400 * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
        + 400 * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
    ) / (
        (48 + 64 * I) * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + (48 - 64 * I) * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        + 100 * exp(2 * T - (58 * X) / 13 + 2)
        + 100 * exp(2 * T + (26 * X) / 29 + 2)
        + exp(4 * T - (1344 * X) / 377 + 4)
        + 400
    )
    pExact = (
        16
        * (
            42688 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 832 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 352 * I * exp(4 * T - (1344 * X) / 377 + 4)
            + 26000 * I * exp(2 * T + (26 * X) / 29 + 2)
            + 34800 * I * exp(2 * T - (58 * X) / 13 + 2)
            - 14384 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 20176 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            - 135 * exp(4 * T - (1344 * X) / 377 + 4)
            + 27300 * exp(2 * T + (26 * X) / 29 + 2)
            + 14500 * exp(2 * T - (58 * X) / 13 + 2)
            + 150800
        )
        * (
            283 * I * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
            - 339 * I * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
            - 8700 * I * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
            - 6500 * I * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
            + 206 * exp((3 - I / 2) * T + 3 - (1513 / 377 + 73 * I / 116) * X)
            + 398 * exp((3 + I / 2) * T + 3 - (503 / 377 + 57 * I / 52) * X)
            - 5800 * exp((1 + I / 2) * T - (29 / 13 + 57 * I / 52) * X + 1)
            - 2600 * exp((1 - I / 2) * T + (13 / 29 - 73 * I / 116) * X + 1)
        )
        / (
            142129
            * (
                64 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
                - 64 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
                + 100 * exp(2 * T - (58 * X) / 13 + 2)
                + 48 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
                + 48 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
                + 100 * exp(2 * T + (26 * X) / 29 + 2)
                + exp(4 * T - (1344 * X) / 377 + 4)
                + 400
            )
            ** 2
        )
    )
    etaExact = (
        (675584 + 2316288 * I)
        * exp((4 + 2 * I) * T + 4 - (1344 / 377 + 352 * I / 377) * X)
        - (3222400 + 6003200 * I)
        * exp((4 - I) * T + 4 + (-334 / 377 + 176 * I / 377) * X)
        + (9843200 + 14182400 * I)
        * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
        - (4860800 + 4774400 * I)
        * exp((4 - I) * T + 4 + (-2354 / 377 + 176 * I / 377) * X)
        - (40928 + 13696 * I) * exp((6 + I) * T + 6 - (2016 / 377 + 176 * I / 377) * X)
        + (-4860800 + 4774400 * I)
        * exp((4 + I) * T + 4 - (2354 / 377 + 176 * I / 377) * X)
        + (-3222400 + 6003200 * I)
        * exp((4 + I) * T + 4 - (334 / 377 + 176 * I / 377) * X)
        + (675584 - 2316288 * I)
        * exp((4 - 2 * I) * T + 4 + (-1344 / 377 + 352 * I / 377) * X)
        + (9843200 - 14182400 * I)
        * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
        + (-40928 + 13696 * I)
        * exp((6 - I) * T + 6 + (-2016 / 377 + 176 * I / 377) * X)
        - 377 * exp(8 * T - (2688 * X) / 377 + 8)
        + 6960000 * exp(2 * T - (58 * X) / 13 + 2)
        - 13520000 * exp(2 * T + (26 * X) / 29 + 2)
        - 3770000 * exp(4 * T + (52 * X) / 29 + 4)
        - 3770000 * exp(4 * T - (116 * X) / 13 + 4)
        - 13844800 * exp(4 * T - (1344 * X) / 377 + 4)
        - 33800 * exp(6 * T - (3026 * X) / 377 + 6)
        + 17400 * exp(6 * T - (1006 * X) / 377 + 6)
        - 60320000
    ) / (
        377
        * (
            64 * I * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            - 64 * I * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * exp(2 * T - (58 * X) / 13 + 2)
            + 48 * exp((2 - I) * T + 2 + (-672 / 377 + 176 * I / 377) * X)
            + 48 * exp((2 + I) * T + 2 - (672 / 377 + 176 * I / 377) * X)
            + 100 * exp(2 * T + (26 * X) / 29 + 2)
            + exp(4 * T - (1344 * X) / 377 + 4)
            + 400
        )
        ** 2
    )
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
# print(X_u_train,'\n', etau_icbc)
# exit()
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

RAR = False
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

LBFGS = False
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
# lala=X.flatten()[:, None]#(51456,1) ，改成[:]和 都会是(51456,)，flatten要变成矩阵只能是None，0:1之类的会报错

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
    plt.plot(x, H_exact[index, :], "b-", linewidth=2, label="Exact")
    plt.plot(x, H_pred[index, :], "r--", linewidth=2, label="Prediction")
    ax.set_ylabel(f"$|{name}(t,z)|$")
    ax.set_xlabel("$z$")
    plt.title("t=%s" % tt0)
    plt.legend()
    ax = plt.subplot(2, 1, 2)

    index = round((tt1 - t_lower) / A * (nt - 1))
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
