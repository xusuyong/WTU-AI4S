import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh, sqrt, arctan

# import torch
import re
import scipy.io
import time

start_time = time.time()
# from deepxde.backend import tf

x_lower = -3
x_upper = 10
t_lower = -5
t_upper = 5

period = 1
A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -45
dpi = 130
nx = 512
nt = 512
x = np.linspace(x_lower, x_upper, nx)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
I = 1j


def solution_a(XT):
    x = XT[:, 0:1]
    t = XT[:, 1:2]

    phi1Exact = (
        (
            (1 / 4 + I) * exp(((2 * x / sqrt(t**2 + 1) - 1) + I) + 4 * I * arctan(t))
            + (1 / 2 + I) * exp(((x / sqrt(t**2 + 1) - 3) - I) + arctan(t) * I)
            + (425 / 73728 + 425 * I / 36864)
            * exp(((5 * x / sqrt(t**2 + 1) - 5) - I) + arctan(t) * I)
            + (125 / 9216 + 125 * I / 2304)
            * exp(((4 * x / sqrt(t**2 + 1) - 7) + I) + 4 * I * arctan(t))
        )
        * exp(t * x**2 * I / (2 * (2 * t**2 + 2)) - log(t**2 + 1) / 2)
        / (
            1
            + (425 * exp(4 * x / sqrt(t**2 + 1) - 2)) / 4096
            + (125 * exp(2 * x / sqrt(t**2 + 1) - 6)) / 256
            + (225 / 2048 + 25 * I / 1024)
            * exp(3 * x / sqrt(t**2 + 1) - 4 + 2 * I + 3 * I * arctan(t))
            + (25 / 128 - 25 * I / 576)
            * exp(3 * x / sqrt(t**2 + 1) - 4 - 2 * I - 3 * I * arctan(t))
            + (53125 * exp(6 * x / sqrt(t**2 + 1) - 8)) / 150994944
        )
    )

    phim1Exact = (
        (
            -(1 / 16 + I / 4)
            * exp(((2 * x / sqrt(t**2 + 1) - 1) + I) + 4 * I * arctan(t))
            - (1 / 8 + I / 4) * exp(((x / sqrt(t**2 + 1) - 3) - I) + arctan(t) * I)
            - (425 / 294912 + 425 * I / 147456)
            * exp(((5 * x / sqrt(t**2 + 1) - 5) - I) + arctan(t) * I)
            - (125 / 36864 + 125 * I / 9216)
            * exp(((4 * x / sqrt(t**2 + 1) - 7) + I) + 4 * I * arctan(t))
        )
        * exp(t * x**2 * I / (2 * (2 * t**2 + 2)) - log(t**2 + 1) / 2)
        / (
            1
            + (425 * exp(4 * x / sqrt(t**2 + 1) - 2)) / 4096
            + (125 * exp(2 * x / sqrt(t**2 + 1) - 6)) / 256
            + (225 / 2048 + 25 * I / 1024)
            * exp(3 * x / sqrt(t**2 + 1) - 4 + 2 * I + 3 * I * arctan(t))
            + (25 / 128 - 25 * I / 576)
            * exp(3 * x / sqrt(t**2 + 1) - 4 - 2 * I - 3 * I * arctan(t))
            + (53125 * exp(6 * x / sqrt(t**2 + 1) - 8)) / 150994944
        )
    )

    phi0Exact = (
        (
            (1 / 2 - I / 8)
            * exp(((2 * x / sqrt(t**2 + 1) - 1) + I) + 4 * I * arctan(t))
            + (1 / 2 - I / 4) * exp(((x / sqrt(t**2 + 1) - 3) - I) + arctan(t) * I)
            + (425 / 73728 - 425 * I / 147456)
            * exp(((5 * x / sqrt(t**2 + 1) - 5) - I) + arctan(t) * I)
            + (125 / 4608 - 125 * I / 18432)
            * exp(((4 * x / sqrt(t**2 + 1) - 7) + I) + 4 * I * arctan(t))
        )
        * exp(t * x**2 * I / (2 * (2 * t**2 + 2)) - log(t**2 + 1) / 2)
        / (
            1
            + (425 * exp(4 * x / sqrt(t**2 + 1) - 2)) / 4096
            + (125 * exp(2 * x / sqrt(t**2 + 1) - 6)) / 256
            + (225 / 2048 + 25 * I / 1024)
            * exp(3 * x / sqrt(t**2 + 1) - 4 + 2 * I + 3 * I * arctan(t))
            + (25 / 128 - 25 * I / 576)
            * exp(3 * x / sqrt(t**2 + 1) - 4 - 2 * I - 3 * I * arctan(t))
            + (53125 * exp(6 * x / sqrt(t**2 + 1) - 8)) / 150994944
        )
    )

    return (
        real(phi1Exact),
        imag(phi1Exact),
        real(phim1Exact),
        imag(phim1Exact),
        real(phi0Exact),
        imag(phi0Exact),
    )


(
    phi1Exact_u,
    phi1Exact_v,
    phim1Exact_u,
    phim1Exact_v,
    phi0Exact_u,
    phi0Exact_v,
) = solution_a(X_star)

phi1h_true = np.sqrt(phi1Exact_u**2 + phi1Exact_v**2).flatten()
phim1h_true = np.sqrt(phim1Exact_u**2 + phim1Exact_v**2).flatten()
phi0h_true = np.sqrt(phi0Exact_u**2 + phi0Exact_v**2).flatten()

phi1Exact_h = phi1h_true.reshape(X.shape[0], X.shape[1])
phim1Exact_h = phim1h_true.reshape(X.shape[0], X.shape[1])
phi0Exact_h = phi0h_true.reshape(X.shape[0], X.shape[1])

fig8 = plt.figure(
    "3d真解phi_{+1}", dpi=dpi, facecolor=None, edgecolor=None, layout="constrained"
)
ax = fig8.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    phi1Exact_h**2,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="rainbow",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
    linewidth=0,
    antialiased=False,
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|\phi_{+1}|^2$")
ax.set_zlim(0, 4)
plt.locator_params(nbins=5)
# fig8.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
# plt.tight_layout()#自动调整大小和间距，使各个子图标签不重叠
plt.savefig("3dphi1.png", dpi=250)

fig9 = plt.figure("3d真解phi_{-1}", dpi=dpi, facecolor=None, edgecolor=None)
ax = fig9.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    phim1Exact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="jet",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|\phi_{-1}(t,x)|$")
# fig9.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
plt.tight_layout()  # 自动调整大小和间距，使各个子图标签不重叠
plt.savefig("3dphim1.png", dpi=250)

fig10 = plt.figure("3d真解phi_0", dpi=dpi, facecolor=None, edgecolor=None)
ax = fig10.add_subplot(projection="3d")
surf = ax.plot_surface(
    X,
    T,
    phi0Exact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="jet",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$|\phi_0(t,x)|$")
# fig10.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elevation, azimuth)
plt.tight_layout()  # 自动调整大小和间距，使各个子图标签不重叠
plt.savefig("3dphi0.png", dpi=250)

plt.figure(dpi=250)
plt.imshow(
    phi1Exact_h,
    extent=[x_lower, x_upper, t_lower, t_upper],
    cmap="jet",
    # norm=norm1,
    origin="lower",  # 这个有影响，不要注释掉
    aspect="auto",
)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()
plt.savefig("phi1.png", dpi=250)

plt.figure(dpi=250)
plt.imshow(
    phim1Exact_h,
    extent=[x_lower, x_upper, t_lower, t_upper],
    cmap="jet",
    # norm=norm1,
    origin="lower",
    aspect="auto",
)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()
plt.savefig("phim1.png", dpi=250)

plt.figure(dpi=250)
plt.imshow(
    phi0Exact_h,
    extent=[x_lower, x_upper, t_lower, t_upper],
    cmap="jet",
    # norm=norm1,
    origin="lower",  # 这个有影响，不要注释掉
    aspect="auto",
)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar()
plt.savefig("phi0.png", dpi=250)

# plt.show()
