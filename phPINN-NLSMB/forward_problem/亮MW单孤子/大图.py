import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import matplotlib

# %%
<<<<<<< Updated upstream
folder_name = "output_2023年12月25日14时26分16秒"
file0 = f"{folder_name}/loss.dat"
data1 = io.loadmat(f"{folder_name}/预测结果_亮MW单孤子.mat")
=======
file0 = r"D:\PyCharm 2023.3.1\demo\WTU-4S\phPINN-NLSMB\forward_problem\亮MW单孤子\output_2023年12月25日21时15分44秒\loss.dat"
data1 = io.loadmat(
    r"D:\PyCharm 2023.3.1\demo\WTU-4S\phPINN-NLSMB\forward_problem\亮MW单孤子\output_2023年12月25日21时15分44秒\预测结果_亮MW单孤子.mat"
)
>>>>>>> Stashed changes
tt0 = -2
tt1 = 2
x = data1["x"].flatten()
t = data1["t"].flatten()
z_lower = np.min(x)
z_upper = np.max(x)
t_lower = np.min(t)
t_upper = np.max(t)
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
elapsed = data1["elapsed"][0, 0]
E_L2_relative_error = data1["E_L2_relative_error"][0, 0]
p_L2_relative_error = data1["p_L2_relative_error"][0, 0]
eta_L2_relative_error = data1["eta_L2_relative_error"][0, 0]
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
dpi = 300
aaa = 9
cmap0 = "jet"  # PuOr seismic pink_r jet coolwarm summer ocean

# %% 新画图
fig = plt.figure(layout="constrained", figsize=(aaa, aaa / 2.5 * 3), dpi=dpi)

# fig.suptitle('fig')

subfigs = fig.subfigures(
    3,
    2,
    # wspace=0.07
)
# %% 第一个子图Exact Dynamics
subfigs[0, 0].set_facecolor("0.97")
subfigs[0, 0].suptitle("Exact Dynamics")
ax00 = subfigs[0, 0].subplots(3, 1, sharex=True)
ax00[0].set_ylabel("$|E(t,z)|$", fontsize="medium")
h0 = ax00[0].imshow(
    EExact_h.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h0, ax=ax00[0])
# ax00[0].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=3,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax00[0].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax00[0].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax00[1].set_ylabel("$|p(t,z)|$", fontsize="medium")
h1 = ax00[1].imshow(
    pExact_h.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# plt.colorbar(h1, ax=ax00[1])
# ax00[1].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=3,clip_on=False)
# line = np.linspace(z_lower, z_upper, 2)[:,None]
ax00[1].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax00[1].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
ax00[2].set_ylabel("$|\eta(t,z)|$", fontsize="medium")
h2 = ax00[2].imshow(
    etaExact_h.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
plt.colorbar(h0, ax=ax00, shrink=0.9)
# ax00[2].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=3,clip_on=False)
# line = np.linspace(z_lower, z_upper, 2)[:,None]
ax00[2].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax00[2].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax00[2].legend(loc='center', bbox_to_anchor=(0.6, -0.3),borderaxespad = 0., fontsize=12,frameon=False)#bbox_to_anchor的坐标决定loc那个点的位置
# %% 第二个子图Prediction Dynamics
subfigs[0, 1].set_facecolor("0.97")
subfigs[0, 1].suptitle("Prediction Dynamics")
# fig15 = plt.figure('平面预测演化图', dpi=dpi)
# plt.suptitle("Prediction Dynamics")
ax01 = subfigs[0, 1].subplots(3, 1, sharex=True)
# ax3.set_title("Prediction Dynamics")
ax01[0].set_ylabel("$|E(t,z)|$", fontsize="medium")
h3 = ax01[0].imshow(
    EH_pred.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# # ax01[0].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=1,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax01[0].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax01[0].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax01[1] = plt.subplot(6,3,5)
ax01[1].set_ylabel("$|p(t,z)|$", fontsize="medium")
h4 = ax01[1].imshow(
    pH_pred.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
# # ax01[1].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=1,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax01[1].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax01[1].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# ax01[2] = plt.subplot(6,3,8)
ax01[2].set_ylabel("$|\eta(t,z)|$", fontsize="medium")
h5 = ax01[2].imshow(
    etaH_pred.T,
    interpolation="nearest",
    cmap=cmap0,
    extent=[t_lower, t_upper, z_lower, z_upper],
    norm=norm0,
    origin="lower",
    aspect="auto",
)
plt.colorbar(h0, ax=ax01, shrink=0.9)
# # ax01[2].plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=1,clip_on=False)
line = np.linspace(z_lower, z_upper, 2)[:, None]
ax01[2].plot(tt0 * np.ones((2, 1)), line, "k--", linewidth=1)
ax01[2].plot(tt1 * np.ones((2, 1)), line, "k--", linewidth=1)
# %%第三个子图t=-2
subfigs[1, 0].set_facecolor("0.97")
subfigs[1, 0].suptitle("t=%s" % tt0)
ax10 = subfigs[1, 0].subplots()
index = round(
    (tt0 - t_lower) / A * (nt - 1)
)  # index只能是0-200（总共有201行,=0时索引第1个数,=200时索引第201）
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

# %%第四个子图t=2
subfigs[1, 1].set_facecolor("0.97")
subfigs[1, 1].suptitle("t=%s" % tt1)
ax11 = subfigs[1, 1].subplots()
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
# Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
# %%第五个子图Loss
subfigs[2, 0].set_facecolor("0.97")
# subfigs[2,0].suptitle("Loss")
data = np.loadtxt(file0, skiprows=1)
iterations = data[:, 0]
loss_len = (data.shape[1] - 1) // 2
loss_train = np.sum(data[:, 1 : 1 + loss_len], axis=1)
loss_test = np.sum(data[:, 1 + loss_len : 1 + loss_len + loss_len], axis=1)

print("last train loss: %.6e" % loss_train[-1], "iterations: %s" % iterations[-1])
print("last test loss: %.6e" % loss_test[-1])
ax20 = subfigs[2, 0].subplots()
# ax=fig16.add_subplot(3,2,4)
plt.plot(iterations[:301], loss_train[:301], linewidth=1, label="Adam")
plt.plot(iterations[300:], loss_train[300:], linewidth=1, label="L-BFGS")
# plt.plot(iterations, loss_test, 'b-', linewidth=1, label='Test loss')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend(frameon=False, loc="upper right")
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


<<<<<<< Updated upstream
plt.savefig(f"{folder_name}/亮MW.pdf", dpi="figure")
=======
plt.savefig("output_2023年12月25日21时15分44秒/亮MW.pdf", dpi="figure")
>>>>>>> Stashed changes
plt.show()
