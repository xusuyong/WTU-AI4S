import numpy as np
import scipy.io  # python读取.mat数据之scipy.io&h5py
import matplotlib.pyplot as plt
import matplotlib
import csv  # 导入csv模块

# %%
file0 = r"F:\QQ文件\NLS-MB项目\dde\文献75亮亮暗单孤子\训练数据\loss.dat"
data1 = scipy.io.loadmat(r"F:\QQ文件\NLS-MB项目\dde\文献75亮亮暗单孤子\训练数据\预测结果_亮亮暗.mat")
tt0 = -2
tt1 = 2
x = data1["x"]
t = data1["t"]
z_lower = float(x[0])
z_upper = float(x[-1])
t_lower = float(t[0])
t_upper = float(t[-1])
nx = x.shape[0]
nt = t.shape[0]
X, T = np.meshgrid(x, t)
EExact_h = data1["EExact_h"]
pExact_h = data1["pExact_h"]
etaExact_h = data1["etaExact_h"]
EH_pred = data1["EH_pred"]
pH_pred = data1["pH_pred"]
etaH_pred = data1["etaH_pred"]
# X_u_train = data1['X_u_train']
elapsed = data1["elapsed"]
E_L2_relative_error = float(data1["E_L2_relative_error"])
p_L2_relative_error = float(data1["p_L2_relative_error"])
eta_L2_relative_error = float(data1["eta_L2_relative_error"])
print("x", z_lower, z_upper, "  t", t_lower, t_upper)
print("Training time: %.4fs or %.4fminutes" % (elapsed, elapsed / 60))
print(
    "E_L2_relative_error: %.6e,\np_L2_relative_error: %.6e,\neta_L2_relative_error: %.6e"
    % (E_L2_relative_error, p_L2_relative_error, eta_L2_relative_error)
)
# norm0 = None
norm0 = matplotlib.colors.Normalize(
    vmin=np.min([EExact_h, pExact_h, etaExact_h, EH_pred, pH_pred, etaH_pred]),
    vmax=np.max([EExact_h, pExact_h, etaExact_h, EH_pred, pH_pred, etaH_pred]),
)
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
vmax = np.max(
    [
        np.abs(EExact_h - EH_pred),
        np.abs(pExact_h - pH_pred),
        np.abs(etaExact_h - etaH_pred),
    ]
)
print("vmax=%.5f" % vmax)
A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -40
dpi = 130
aaa = 9
cmap0 = "jet"  # PuOr seismic pink_r jet coolwarm summer ocean

# %% 新画图
fig = plt.figure(layout="constrained", figsize=(aaa, aaa / 2.5 * 3))

# fig.suptitle('fig')

subfigs = fig.subfigures(
    3,
    2,
    # wspace=0.07
)
# %% 第一个子图Exact Dynamics
subfigs[0, 0].set_facecolor("0.97")
# subfigs[0,0].suptitle("Exact Dynamics")
ax00 = subfigs[0, 0].add_subplot(projection="3d")
# fig9 = plt.figure("3d真解p", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig9.add_subplot(projection='3d')
surf = ax00.plot_surface(
    X,
    T,
    EExact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="coolwarm",  # 设置颜色映射 还可以设置成YlGnBu_r和viridis
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax00.grid(False)#关闭背景的网格线
ax00.set_xlabel("$z$")
ax00.set_ylabel("$t$")
ax00.set_zlabel("$|E(t,z)|$")
# fig8.colorbar(surf, shrink=0.5, aspect=5)
ax00.view_init(elevation, azimuth)
# plt.tight_layout()
# ax00[2].legend(loc='center', bbox_to_anchor=(0.6, -0.3),borderaxespad = 0., fontsize=12,frameon=False)#bbox_to_anchor的坐标决定loc那个点的位置
# %% 第二个子图Prediction Dynamics
subfigs[0, 1].set_facecolor("0.97")
ax00 = subfigs[0, 1].add_subplot(projection="3d")
# fig9 = plt.figure("3d真解p", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig9.add_subplot(projection='3d')
surf = ax00.plot_surface(
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
ax00.set_xlabel("$z$")
ax00.set_ylabel("$t$")
ax00.set_zlabel("$|p(t,z)|$")
# fig9.colorbar(surf, shrink=0.5, aspect=5)
ax00.view_init(elevation, azimuth)
# %%第三个子图t=-2
subfigs[1, 0].set_facecolor("0.97")
ax00 = subfigs[1, 0].add_subplot(projection="3d")
# fig9 = plt.figure("3d真解p", dpi=dpi, facecolor=None, edgecolor=None)
# ax = fig9.add_subplot(projection='3d')
surf = ax00.plot_surface(
    X,
    T,
    etaExact_h,
    rstride=stride,  # 指定行的跨度
    cstride=stride,  # 指定列的跨度
    cmap="coolwarm",  # 设置颜色映射
    linewidth=0,
    antialiased=False,
)  # 抗锯齿
# ax00.grid(False)#关闭背景的网格线
ax00.set_xlabel("$z$")
ax00.set_ylabel("$t$")
ax00.set_zlabel("$|\eta(t,z)|$")
# fig10.colorbar(surf, shrink=0.5, aspect=5)
ax00.view_init(elevation, azimuth)
# %%第四个子图t=2
subfigs[1, 1].set_facecolor("0.97")
subfigs[1, 1].suptitle("t=%s" % tt0)
ax11 = subfigs[1, 1].subplots()
index = round((tt0 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(
    x,
    EExact_h[index, :],
    x,
    EH_pred[index, :],
    "--",
    x,
    pExact_h[index, :],
    x,
    pH_pred[index, :],
    "--",
    x,
    etaExact_h[index, :],
    x,
    etaH_pred[index, :],
    "y--",
    linewidth=2,
)
plt.xlabel("$z$")
plt.ylabel("Amplitude")
plt.legend(
    [
        "$E$ exact",
        "$E$ prediction",
        "$p$ exact",
        "$p$ prediction",
        "$\eta$ exact",
        "$\eta$ prediction",
    ],
    fontsize="x-small",
    frameon=False,
)
# Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
# %%第五个子图Loss
subfigs[2, 0].set_facecolor("0.97")
# subfigs[2,0].suptitle("Loss")
subfigs[2, 0].set_facecolor("0.97")
subfigs[2, 0].suptitle("t=%s" % tt1)
ax11 = subfigs[2, 0].subplots()
index = round((tt1 - t_lower) / A * (nt - 1))  # 只能是0-200（总共有201行）
plt.plot(
    x,
    EExact_h[index, :],
    x,
    EH_pred[index, :],
    "--",
    x,
    pExact_h[index, :],
    x,
    pH_pred[index, :],
    "--",
    x,
    etaExact_h[index, :],
    x,
    etaH_pred[index, :],
    "y--",
    linewidth=2,
)
plt.xlabel("$z$")
plt.ylabel("Amplitude")
plt.legend(
    [
        "$E$ exact",
        "$E$ prediction",
        "$p$ exact",
        "$p$ prediction",
        "$\eta$ exact",
        "$\eta$ prediction",
    ],
    fontsize="x-small",
    frameon=False,
)

# %%第六个子图Absolute error
subfigs[2, 1].set_facecolor("0.97")
subfigs[2, 1].suptitle("Absolute error")
cmap1 = "ocean_r"  # PuOr seismic pink_r jet coolwarm summer ocean spring cool
ax21 = subfigs[2, 1].subplots(3, 1, sharex=True)
# ax0.set_title("Absolute error")
ax21[0].set_ylabel("$|E(t,z)|$", fontsize="medium")
h0 = ax21[0].imshow(
    np.abs(EExact_h - EH_pred).T,
    interpolation="nearest",
    cmap=cmap1,  # PuOr
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax21[0])
# ax21[1] = plt.subplot(6,3,6)
ax21[1].set_ylabel("$|p(t,z)|$", fontsize="medium")
h1 = ax21[1].imshow(
    np.abs(pExact_h - pH_pred).T,
    interpolation="nearest",
    cmap=cmap1,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h, ax=ax21[1])
# ax21[2] = plt.subplot(6,3,9)
ax21[2].set_ylabel("$|\eta(t,z)|$", fontsize="medium")
h2 = ax21[2].imshow(
    np.abs(etaExact_h - etaH_pred).T,
    interpolation="nearest",
    cmap=cmap1,  # seismic pink_r
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm1,
    origin="lower",
    aspect="auto",
)
plt.colorbar(h0, ax=ax21, shrink=0.9)
subfigs[0, 0].text(
    0, 1, "A", fontsize=18, style="normal", ha="left", va="top", wrap=True
)
subfigs[0, 1].text(
    0, 1, "B", fontsize=18, style="normal", ha="left", va="top", wrap=True
)
subfigs[1, 0].text(
    0, 1, "C", fontsize=18, style="normal", ha="left", va="top", wrap=True
)
subfigs[1, 1].text(
    0, 1, "D", fontsize=18, style="normal", ha="left", va="top", wrap=True
)
subfigs[2, 0].text(
    0, 1, "E", fontsize=18, style="normal", ha="left", va="top", wrap=True
)
subfigs[2, 1].text(
    0, 1, "F", fontsize=18, style="normal", ha="left", va="top", wrap=True
)


plt.savefig(r"亮亮暗加3d")
plt.show()
