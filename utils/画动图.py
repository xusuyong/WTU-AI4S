import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.animation as animation
import numpy as np
from numpy import exp, cos, sin, log, tanh, cosh, real, imag, sinh, sqrt, arctan

Pi = np.pi
x_lower = -15
x_upper = 15
y_lower = -15
y_upper = 15
t_lower = -2
t_upper = 20
period = 1
A = y_upper - y_lower
stride = 2
elevation = 20
azimuth = -40
nx = 256
ny = 256
nt = 128
x = np.linspace(x_lower, x_upper, nx)[:, None]
y = np.linspace(y_lower, y_upper, ny)[:, None]
t = np.linspace(t_lower, t_upper, nt)[:, None]
X, Y = np.meshgrid(x, y)
X1, Y1, T1 = np.meshgrid(x, y, t)
# X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
I = 1j
sech = lambda x: 1 / cosh(x)


def solution2(x1, y1, t1):
    # x = XT[:, 0:1]
    # y = XT[:, 1:2]
    # t = XT[:, 2:3]
    exact2 = (
        (2 + I)
        * exp(
            (
                (1960 * I * exp(-sqrt(2) * t1 / 5) - 4 * t1 + (x1**2 + y1**2) * I)
                * sqrt(2)
            )
            / 40
        )
        * (
            1
            + exp(
                1
                + 49
                * I
                * (1 + (-1) ** (2 / 5))
                * (-5 + sqrt(5))
                * exp(-sqrt(2) * t1 / 5)
                * sqrt(2)
                / (4 * (-1 + (-1) ** (2 / 5)))
                + 2 * I * Pi / 5
                - sqrt(254) * exp(-sqrt(2) * t1 / 10) * x1 / sqrt(15 + 3 * sqrt(5))
                + sqrt(30 - 6 * sqrt(5)) * exp(-sqrt(2) * t1 / 10) * y1 / 3
            )
            - exp(
                1
                + 147
                * (-1) ** (5 / 6)
                * exp(-sqrt(2) * t1 / 5)
                * sqrt(2)
                / (2 * (-1 + (-1) ** (2 / 3)))
                - 2 * I * Pi / 3
                + exp(-sqrt(2) * t1 / 10) * (-sqrt(635) * x1 / 5 + 2 * y1)
                + Pi * I
            )
            - exp(
                (
                    49
                    * sqrt(2)
                    * (
                        (6 + 2 * I * sqrt(3)) / (I + sqrt(3))
                        - (-1) ** (2 / 5)
                        * (-5 + sqrt(5))
                        / (-I + (-1) ** (1 / 10) - (-1) ** (3 / 10) + (-1) ** (7 / 10))
                    )
                    * exp(-sqrt(2) * t1 / 5)
                )
                / 4
                + 2
                - 4 * I * Pi / 15
                + exp(-sqrt(2) * t1 / 10)
                * (
                    -4 * (3 * sqrt(635) + 5 * sqrt(762) / sqrt(5 + sqrt(5))) * x1
                    + 20 * (6 + sqrt(30 - 6 * sqrt(5))) * y1
                )
                / 60
                + Pi * I
            )
            * (
                -1029
                + 147 * sqrt(5)
                + 40 * sqrt(30 - 6 * sqrt(5))
                - 127 * sqrt(30 + 6 * sqrt(5))
                + 127 * sqrt(150 + 30 * sqrt(5))
                + 588 * cos(Pi / 15)
            )
            / (
                1029
                - 147 * sqrt(5)
                + 40 * sqrt(30 - 6 * sqrt(5))
                - 127 * sqrt(30 + 6 * sqrt(5))
                + 127 * sqrt(150 + 30 * sqrt(5))
                + 588 * sin((7 * Pi) / 30)
            )
        )
        / (
            1
            + exp(
                1
                + 49
                * I
                * (1 + (-1) ** (2 / 5))
                * (-5 + sqrt(5))
                * exp(-sqrt(2) * t1 / 5)
                * sqrt(2)
                / (4 * (-1 + (-1) ** (2 / 5)))
                - sqrt(254) * exp(-sqrt(2) * t1 / 10) * x1 / sqrt(15 + 3 * sqrt(5))
                + sqrt(30 - 6 * sqrt(5)) * exp(-sqrt(2) * t1 / 10) * y1 / 3
            )
            + exp(
                1
                + 147
                * (-1) ** (5 / 6)
                * exp(-sqrt(2) * t1 / 5)
                * sqrt(2)
                / (2 * (-1 + (-1) ** (2 / 3)))
                + exp(-sqrt(2) * t1 / 10) * (-sqrt(635) * x1 / 5 + 2 * y1)
            )
            + exp(
                2
                + (
                    49
                    * sqrt(2)
                    * (
                        (6 + 2 * I * sqrt(3)) / (I + sqrt(3))
                        - (-1) ** (2 / 5)
                        * (-5 + sqrt(5))
                        / (-I + (-1) ** (1 / 10) - (-1) ** (3 / 10) + (-1) ** (7 / 10))
                    )
                    * exp(-sqrt(2) * t1 / 5)
                )
                / 4
                + exp(-sqrt(2) * t1 / 10)
                * (
                    -(3 * sqrt(635) + 5 * sqrt(762) / sqrt(5 + sqrt(5))) * x1
                    + 5 * (6 + sqrt(30 - 6 * sqrt(5))) * y1
                )
                / 15
            )
            * (
                -1029
                + 147 * sqrt(5)
                + 40 * sqrt(30 - 6 * sqrt(5))
                - 127 * sqrt(30 + 6 * sqrt(5))
                + 127 * sqrt(150 + 30 * sqrt(5))
                + 588 * cos(Pi / 15)
            )
            / (
                1029
                - 147 * sqrt(5)
                + 40 * sqrt(30 - 6 * sqrt(5))
                - 127 * sqrt(30 + 6 * sqrt(5))
                + 127 * sqrt(150 + 30 * sqrt(5))
                + 588 * sin((7 * Pi) / 30)
            )
        )
    )

    return abs(exact2)


phi1Exact = solution2(X1, Y1, T1)
norm0 = matplotlib.colors.Normalize(vmin=np.min(phi1Exact), vmax=np.max(phi1Exact))

fig, ax = plt.subplots()
ims = []
for i, t_value in enumerate(t):
    phi1Exact = solution2(X, Y, t_value)
    # psi=phi1Exact[..., i]
    im = ax.imshow(phi1Exact, animated=True, cmap="jet",norm=norm0,extent=[x_lower, x_upper, y_lower, y_upper],aspect="auto")
    if i == 0:
        im=ax.imshow(phi1Exact, cmap="jet",norm=norm0,extent=[x_lower, x_upper, y_lower, y_upper],aspect="auto")  # show an initial one first
        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("y")
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

ani.save("平面动图.gif", dpi="figure")


# fig = plt.figure(dpi=250, layout="tight")
# ax = fig.add_subplot(projection="3d")
# ims = []
# for i, t_value in enumerate(t):
#     phi1Exact = solution2(X, Y, t_value)
#     im = ax.plot_surface(
#         X,
#         Y,
#         phi1Exact,
#         animated=True,
#         rstride=2,  # 指定行的跨度
#         cstride=2,  # 指定列的跨度
#         cmap="jet",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
#         linewidth=0,
#         antialiased=False,
#     )
#     # 抗锯齿
#     # if i == 0:
#     #     ax.plot_surface(
#     #         X,
#     #         Y,
#     #         phi1Exact,
#     #         # animated=True,
#     #         rstride=stride,  # 指定行的跨度
#     #         cstride=stride,  # 指定列的跨度
#     #         cmap="jet",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
#     #         linewidth=0,
#     #         antialiased=False,
#     #     )  # 抗锯齿
#     ims.append([im])
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.set_zlabel("$|\phi_{+1}|$")
# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
# ani.save("3维动图2.gif", dpi="figure")

# plt.show()
