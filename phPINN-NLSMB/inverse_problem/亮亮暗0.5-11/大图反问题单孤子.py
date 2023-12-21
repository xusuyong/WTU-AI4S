import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import matplotlib
import re


# %%
file0 = "0噪声1秒/loss.dat"
file1 = "1噪声1秒/loss.dat"
file3 = "3噪声1秒/loss.dat"
file5 = "5噪声1秒/loss.dat"
file10 = "10噪声1秒/loss.dat"

fnamevar0 = "0噪声1秒/variables1.dat"
fnamevar1 = "1噪声1秒/variables1.dat"
fnamevar3 = "3噪声1秒/variables1.dat"
fnamevar5 = "5噪声1秒/variables1.dat"
fnamevar10 = "10噪声1秒/variables1.dat"


data1 = io.loadmat(
    r"D:\xsy\PycharmProjects\WTU-AI4S\phPINN-NLSMB\forward_problem\亮MW单孤子\预测结果不要动它_文献75亮MW单孤子.mat"
)
true_a1 = 0.5
true_a2 = -1
true_o0 = 1
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
X_u_train = data1["X_u_train"]
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

A = t_upper - t_lower
stride = 5
elevation = 20
azimuth = -40
dpi = 130
aaa = 9
cmap0 = "coolwarm"  # PuOr seismic pink_r jet coolwarm summer ocean

# %% 新画图
fig = plt.figure(layout="constrained", figsize=(aaa, aaa / 2.5 * 3))

# fig.suptitle('fig')

subfigs = fig.subfigures(
    3,
    2,
    # wspace=0.07
)
# %% 第一个子图
subfigs[0, 0].set_facecolor("0.96")
# subfigs[0,0].suptitle("Exact Dynamics")
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
ax00[0].plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=1,
    clip_on=False,
)

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
ax00[1].plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=1,
    clip_on=False,
)

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
ax00[2].plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=1,
    clip_on=False,
)

# ax00[2].legend(loc='center', bbox_to_anchor=(0.6, -0.3),borderaxespad = 0., fontsize=12,frameon=False)#bbox_to_anchor的坐标决定loc那个点的位置
# %% 第二个子图Loss
subfigs[0, 1].set_facecolor("0.96")
# subfigs[0,1].suptitle("Loss")
with open(file0, "r", encoding="utf-8") as dat_file:
    iterations, loss_train, loss_test = [], [], []
    for line in dat_file:
        if "#" in line:
            continue
        row = [field.strip() for field in line.split(" ")]
        c1 = float(row[0])
        c2 = float(
            float(row[1])
            + float(row[2])
            + float(row[3])
            + float(row[4])
            + float(row[5])
            + float(row[6])
            + float(row[7])
            + float(row[8])
            + float(row[9])
            + float(row[10])
        )
        c3 = float(
            float(row[11])
            + float(row[12])
            + float(row[13])
            + float(row[14])
            + float(row[15])
            + float(row[16])
            + float(row[17])
            + float(row[18])
            + float(row[19])
            + float(row[20])
        )
        iterations.append(c1)
        loss_train.append(c2)
        loss_test.append(c3)
print("last train loss: %.6e" % c2)
# print('last test loss: %.6e'%c3)
ax20 = subfigs[0, 1].subplots()
plt.plot(iterations, loss_train, linewidth=1, label="nosie=0%")

with open(file1, "r", encoding="utf-8") as dat_file:
    iterations, loss_train, loss_test = [], [], []
    for line in dat_file:
        if "#" in line:
            continue
        row = [field.strip() for field in line.split(" ")]
        c1 = float(row[0])
        c2 = float(
            float(row[1])
            + float(row[2])
            + float(row[3])
            + float(row[4])
            + float(row[5])
            + float(row[6])
            + float(row[7])
            + float(row[8])
            + float(row[9])
            + float(row[10])
        )
        c3 = float(
            float(row[11])
            + float(row[12])
            + float(row[13])
            + float(row[14])
            + float(row[15])
            + float(row[16])
            + float(row[17])
            + float(row[18])
            + float(row[19])
            + float(row[20])
        )
        iterations.append(c1)
        loss_train.append(c2)
        loss_test.append(c3)
print("last train loss: %.6e" % c2)
plt.plot(iterations, loss_train, linewidth=1, label="nosie=1%")

with open(file3, "r", encoding="utf-8") as dat_file:
    iterations, loss_train, loss_test = [], [], []
    for line in dat_file:
        if "#" in line:
            continue
        row = [field.strip() for field in line.split(" ")]
        c1 = float(row[0])
        c2 = float(
            float(row[1])
            + float(row[2])
            + float(row[3])
            + float(row[4])
            + float(row[5])
            + float(row[6])
            + float(row[7])
            + float(row[8])
            + float(row[9])
            + float(row[10])
        )
        c3 = float(
            float(row[11])
            + float(row[12])
            + float(row[13])
            + float(row[14])
            + float(row[15])
            + float(row[16])
            + float(row[17])
            + float(row[18])
            + float(row[19])
            + float(row[20])
        )
        iterations.append(c1)
        loss_train.append(c2)
        loss_test.append(c3)
print("last train loss: %.6e" % c2)
plt.plot(iterations, loss_train, linewidth=1, label="nosie=3%")

with open(file5, "r", encoding="utf-8") as dat_file:
    iterations, loss_train, loss_test = [], [], []
    for line in dat_file:
        if "#" in line:
            continue
        row = [field.strip() for field in line.split(" ")]
        c1 = float(row[0])
        c2 = float(
            float(row[1])
            + float(row[2])
            + float(row[3])
            + float(row[4])
            + float(row[5])
            + float(row[6])
            + float(row[7])
            + float(row[8])
            + float(row[9])
            + float(row[10])
        )
        c3 = float(
            float(row[11])
            + float(row[12])
            + float(row[13])
            + float(row[14])
            + float(row[15])
            + float(row[16])
            + float(row[17])
            + float(row[18])
            + float(row[19])
            + float(row[20])
        )
        iterations.append(c1)
        loss_train.append(c2)
        loss_test.append(c3)
print("last train loss: %.6e" % c2)
plt.plot(iterations, loss_train, linewidth=1, label="nosie=5%")

with open(file10, "r", encoding="utf-8") as dat_file:
    iterations, loss_train, loss_test = [], [], []
    for line in dat_file:
        if "#" in line:
            continue
        row = [field.strip() for field in line.split(" ")]
        c1 = float(row[0])
        c2 = float(
            float(row[1])
            + float(row[2])
            + float(row[3])
            + float(row[4])
            + float(row[5])
            + float(row[6])
            + float(row[7])
            + float(row[8])
            + float(row[9])
            + float(row[10])
        )
        c3 = float(
            float(row[11])
            + float(row[12])
            + float(row[13])
            + float(row[14])
            + float(row[15])
            + float(row[16])
            + float(row[17])
            + float(row[18])
            + float(row[19])
            + float(row[20])
        )
        iterations.append(c1)
        loss_train.append(c2)
        loss_test.append(c3)
print("last train loss: %.6e" % c2)
plt.plot(iterations, loss_train, linewidth=1, label="nosie=10%")

plt.xlabel("iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend(frameon=False, loc="upper right", fontsize="x-small")

# %%第三个子图
subfigs[1, 0].set_facecolor("0.96")
subfigs[1, 0].suptitle("nosie=0")
ax10 = subfigs[1, 0].subplots()
lines = open(fnamevar0, "r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line)),
            sep=",",
        )
        for line in lines
    ]
)
error_0_lambda_1 = abs((true_a1 - Chat[-1, 0]) / true_a1) * 100
error_0_lambda_2 = abs((true_a2 - Chat[-1, 1]) / true_a2) * 100
error_0_omega = abs((true_o0 - Chat[-1, 2]) / true_o0) * 100

print("Error_0 l1: %.5f%%" % (error_0_lambda_1))
print("Error_0 l2: %.5f%%" % (error_0_lambda_2))
print("Error_0 o: %.5f%%" % (error_0_omega))
# fig102 = plt.figure('反问题画图',layout='tight', dpi=130)
# ax = plt.subplot()
l, c = Chat.shape
period = 1
plt.plot(
    range(0, period * l, period), Chat[:, 0], "-", c="#e9963e", label=r"$\alpha_1$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 1], "-", c="#f23b27", label=r"$\alpha_2$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 2], "-", c="#304f9e", label=r"$\omega_0$"
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 0].shape) * true_a1,
    "--",
    c="#e9963e",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 1].shape) * true_a2,
    "--",
    c="#f23b27",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 2].shape) * true_o0,
    "--",
    c="#304f9e",
)
plt.plot(l, Chat[-1, 0], "*", ms="10", c="#e9963e")
ax10.text(l - 5000, Chat[-1, 0] - 0.35, r"$\alpha_1$=%.5f" % Chat[-1, 0], fontsize=10)
plt.plot(l, Chat[-1, 1], "*", ms="10", c="#f23b27")
ax10.text(l - 5000, Chat[-1, 1] + 0.2, r"$\alpha_2$=%.5f" % Chat[-1, 1], fontsize=10)
plt.plot(l, Chat[-1, 2], "*", ms="10", c="#304f9e")
ax10.text(l - 5000, Chat[-1, 2] - 0.3, r"$\omega_0$=%.5f" % Chat[-1, 2], fontsize=10)
square0 = plt.Rectangle(xy=(0, 0.9), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
square1 = plt.Rectangle(xy=(0, 0.45), width=l, height=0.1, fc="#65a9d7", alpha=0.35)
square2 = plt.Rectangle(xy=(0, -1.1), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
ax10.add_patch(square0)  # 把图形加载到绘制区域
ax10.add_patch(square1)  # 把图形加载到绘制区域
ax10.add_patch(square2)  # 把图形加载到绘制区域
plt.legend(
    loc="upper right",
    frameon=False,
    fontsize="small",
)
plt.xlabel("iterations")


# %%第四个子图
# subfigs[1,1].set_facecolor('0.96')
# subfigs[1,1].suptitle("nosie=1%")
# ax11 = subfigs[1,1].subplots()
lines = open(fnamevar1, "r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
error_1_lambda_1 = abs((true_a1 - Chat[-1, 0]) / true_a1) * 100
error_1_lambda_2 = abs((true_a2 - Chat[-1, 1]) / true_a2) * 100
error_1_omega = abs((true_o0 - Chat[-1, 2]) / true_o0) * 100

print("Error_1 l1: %.5f%%" % (error_1_lambda_1))
print("Error_1 l2: %.5f%%" % (error_1_lambda_2))
print("Error_1 o: %.5f%%" % (error_1_omega))
# fig102 = plt.figure('反问题画图',layout='tight', dpi=130)
# ax = plt.subplot()
# l, c = Chat.shape
# period=1
# plt.plot(range(0, period * l, period), Chat[:, 0], "-",c='#e9963e',label=r"$\alpha_1$")
# plt.plot(range(0, period * l, period), Chat[:, 1], "-",c='#f23b27',label=r"$\alpha_2$")
# plt.plot(range(0, period * l, period), Chat[:, 2], "-",c='#304f9e',label=r"$\omega_0$")
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 0].shape) * true_a1, "--",c='#e9963e',)
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 1].shape) * true_a2, "--",c='#f23b27',)
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 2].shape) * true_o0, "--",c='#304f9e',)
# plt.plot(l,Chat[-1,0],'*',ms='10',c='#e9963e');ax11.text(l-5000,Chat[-1,0]-0.35, r'$\alpha_1$=%.5f'%Chat[-1,0], fontsize=10)
# plt.plot(l,Chat[-1,1],'*',ms='10',c='#f23b27');ax11.text(l-5000,Chat[-1,1]+0.2, r'$\alpha_2$=%.5f'%Chat[-1,1], fontsize=10)
# plt.plot(l,Chat[-1,2],'*',ms='10',c='#304f9e');ax11.text(l-5000,Chat[-1,2]-0.3, r'$\omega_0$=%.5f'%Chat[-1,2], fontsize=10)
# square0 = plt.Rectangle(xy=(0, 0.9), width=l, height=0.2, fc='#65a9d7', alpha=0.35)
# square1 = plt.Rectangle(xy=(0, 0.45), width=l, height=0.1,fc='#65a9d7', alpha=0.35)
# square2 = plt.Rectangle(xy=(0, -1.1), width=l, height=0.2,fc='#65a9d7', alpha=0.35)
# ax11.add_patch(square0)  # 把图形加载到绘制区域
# ax11.add_patch(square1)  # 把图形加载到绘制区域
# ax11.add_patch(square2)  # 把图形加载到绘制区域
# plt.legend(
#            loc="upper right",
# frameon=False,
# fontsize='small',
#            )
# plt.xlabel("iterations")

# %%第五个子图
# subfigs[2,0].set_facecolor('0.96')
# subfigs[2,0].suptitle("nosie=3%")
# ax20 = subfigs[2,0].subplots()
lines = open(fnamevar3, "r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
error_3_lambda_1 = abs((true_a1 - Chat[-1, 0]) / true_a1) * 100
error_3_lambda_2 = abs((true_a2 - Chat[-1, 1]) / true_a2) * 100
error_3_omega = abs((true_o0 - Chat[-1, 2]) / true_o0) * 100

print("Error_3 l1: %.5f%%" % (error_3_lambda_1))
print("Error_3 l2: %.5f%%" % (error_3_lambda_2))
print("Error_3 o: %.5f%%" % (error_3_omega))
# fig102 = plt.figure('反问题画图',layout='tight', dpi=130)
# ax = plt.subplot()
# l, c = Chat.shape
# period=1
# plt.plot(range(0, period * l, period), Chat[:, 0], "-",c='#e9963e',label=r"$\alpha_1$")
# plt.plot(range(0, period * l, period), Chat[:, 1], "-",c='#f23b27',label=r"$\alpha_2$")
# plt.plot(range(0, period * l, period), Chat[:, 2], "-",c='#304f9e',label=r"$\omega_0$")
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 0].shape) * true_a1, "--",c='#e9963e',)
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 1].shape) * true_a2, "--",c='#f23b27',)
# plt.plot(range(0, period * l, period), np.ones(Chat[:, 2].shape) * true_o0, "--",c='#304f9e',)
# plt.plot(l,Chat[-1,0],'*',ms='10',c='#e9963e');ax20.text(l-5000,Chat[-1,0]-0.35, r'$\alpha_1$=%.5f'%Chat[-1,0], fontsize=10)
# plt.plot(l,Chat[-1,1],'*',ms='10',c='#f23b27');ax20.text(l-5000,Chat[-1,1]+0.2, r'$\alpha_2$=%.5f'%Chat[-1,1], fontsize=10)
# plt.plot(l,Chat[-1,2],'*',ms='10',c='#304f9e');ax20.text(l-5000,Chat[-1,2]-0.3, r'$\omega_0$=%.5f'%Chat[-1,2], fontsize=10)
# square0 = plt.Rectangle(xy=(0, 0.9), width=l, height=0.2, fc='#65a9d7', alpha=0.35)
# square1 = plt.Rectangle(xy=(0, 0.45), width=l, height=0.1,fc='#65a9d7', alpha=0.35)
# square2 = plt.Rectangle(xy=(0, -1.1), width=l, height=0.2,fc='#65a9d7', alpha=0.35)
# ax20.add_patch(square0)  # 把图形加载到绘制区域
# ax20.add_patch(square1)  # 把图形加载到绘制区域
# ax20.add_patch(square2)  # 把图形加载到绘制区域
# plt.legend(
#            loc="upper right",
# frameon=False,
# fontsize='small',
#            )
# plt.xlabel("iterations")


# %%第六个子图
subfigs[1, 1].set_facecolor("0.96")
subfigs[1, 1].suptitle("nosie=5%")
ax21 = subfigs[1, 1].subplots()
lines = open(fnamevar5, "r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
error_5_lambda_1 = abs((true_a1 - Chat[-1, 0]) / true_a1) * 100
error_5_lambda_2 = abs((true_a2 - Chat[-1, 1]) / true_a2) * 100
error_5_omega = abs((true_o0 - Chat[-1, 2]) / true_o0) * 100

print("Error_5 l1: %.5f%%" % (error_5_lambda_1))
print("Error_5 l2: %.5f%%" % (error_5_lambda_2))
print("Error_5 o: %.5f%%" % (error_5_omega))
# fig102 = plt.figure('反问题画图',layout='tight', dpi=130)
# ax = plt.subplot()
l, c = Chat.shape
period = 1
plt.plot(
    range(0, period * l, period), Chat[:, 0], "-", c="#e9963e", label=r"$\alpha_1$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 1], "-", c="#f23b27", label=r"$\alpha_2$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 2], "-", c="#304f9e", label=r"$\omega_0$"
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 0].shape) * true_a1,
    "--",
    c="#e9963e",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 1].shape) * true_a2,
    "--",
    c="#f23b27",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 2].shape) * true_o0,
    "--",
    c="#304f9e",
)
plt.plot(l, Chat[-1, 0], "*", ms="10", c="#e9963e")
ax21.text(l - 5000, Chat[-1, 0] - 0.35, r"$\alpha_1$=%.5f" % Chat[-1, 0], fontsize=10)
plt.plot(l, Chat[-1, 1], "*", ms="10", c="#f23b27")
ax21.text(l - 5000, Chat[-1, 1] + 0.2, r"$\alpha_2$=%.5f" % Chat[-1, 1], fontsize=10)
plt.plot(l, Chat[-1, 2], "*", ms="10", c="#304f9e")
ax21.text(l - 5000, Chat[-1, 2] - 0.3, r"$\omega_0$=%.5f" % Chat[-1, 2], fontsize=10)
square0 = plt.Rectangle(xy=(0, 0.9), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
square1 = plt.Rectangle(xy=(0, 0.45), width=l, height=0.1, fc="#65a9d7", alpha=0.35)
square2 = plt.Rectangle(xy=(0, -1.1), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
ax21.add_patch(square0)  # 把图形加载到绘制区域
ax21.add_patch(square1)  # 把图形加载到绘制区域
ax21.add_patch(square2)  # 把图形加载到绘制区域
plt.legend(
    loc="upper right",
    frameon=False,
    fontsize="small",
)
plt.xlabel("iterations")

# %%第七个子图
subfigs[2, 0].set_facecolor("0.96")
subfigs[2, 0].suptitle("nosie=10%")
ax30 = subfigs[2, 0].subplots()
lines = open(fnamevar10, "r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
error_10_lambda_1 = abs((true_a1 - Chat[-1, 0]) / true_a1) * 100
error_10_lambda_2 = abs((true_a2 - Chat[-1, 1]) / true_a2) * 100
error_10_omega = abs((true_o0 - Chat[-1, 2]) / true_o0) * 100

print("Error_10 l1: %.5f%%" % (error_10_lambda_1))
print("Error_10 l2: %.5f%%" % (error_10_lambda_2))
print("Error_10 o: %.5f%%" % (error_10_omega))
# fig102 = plt.figure('反问题画图',layout='tight', dpi=130)
# ax = plt.subplot()
l, c = Chat.shape
period = 1
plt.plot(
    range(0, period * l, period), Chat[:, 0], "-", c="#e9963e", label=r"$\alpha_1$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 1], "-", c="#f23b27", label=r"$\alpha_2$"
)
plt.plot(
    range(0, period * l, period), Chat[:, 2], "-", c="#304f9e", label=r"$\omega_0$"
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 0].shape) * true_a1,
    "--",
    c="#e9963e",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 1].shape) * true_a2,
    "--",
    c="#f23b27",
)
plt.plot(
    range(0, period * l, period),
    np.ones(Chat[:, 2].shape) * true_o0,
    "--",
    c="#304f9e",
)
plt.plot(l, Chat[-1, 0], "*", ms="10", c="#e9963e")
ax30.text(l - 5000, Chat[-1, 0] - 0.35, r"$\alpha_1$=%.5f" % Chat[-1, 0], fontsize=10)
plt.plot(l, Chat[-1, 1], "*", ms="10", c="#f23b27")
ax30.text(l - 5000, Chat[-1, 1] + 0.2, r"$\alpha_2$=%.5f" % Chat[-1, 1], fontsize=10)
plt.plot(l, Chat[-1, 2], "*", ms="10", c="#304f9e")
ax30.text(l - 5000, Chat[-1, 2] - 0.3, r"$\omega_0$=%.5f" % Chat[-1, 2], fontsize=10)
square0 = plt.Rectangle(xy=(0, 0.9), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
square1 = plt.Rectangle(xy=(0, 0.45), width=l, height=0.1, fc="#65a9d7", alpha=0.35)
square2 = plt.Rectangle(xy=(0, -1.1), width=l, height=0.2, fc="#65a9d7", alpha=0.35)
ax30.add_patch(square0)  # 把图形加载到绘制区域
ax30.add_patch(square1)  # 把图形加载到绘制区域
ax30.add_patch(square2)  # 把图形加载到绘制区域
plt.legend(
    loc="upper right",
    frameon=False,
    fontsize="small",
)
plt.xlabel("iterations")
# %%第八个子图
subfigs[2, 1].set_facecolor("0.96")
# subfigs[2,1].suptitle("Parameter estimation result")
ax31 = subfigs[2, 1].subplots()
plt.plot(
    [0, 1, 3, 5, 10],
    [
        error_0_lambda_1,
        error_1_lambda_1,
        error_3_lambda_1,
        error_5_lambda_1,
        error_10_lambda_1,
    ],
    "^--",
    c="#e9963e",
    label=r"$\alpha_1$",
)
plt.plot(
    [0, 1, 3, 5, 10],
    [
        error_0_lambda_2,
        error_1_lambda_2,
        error_3_lambda_2,
        error_5_lambda_2,
        error_10_lambda_2,
    ],
    "v--",
    c="#f23b27",
    label=r"$\alpha_2$",
)
plt.plot(
    [0, 1, 3, 5, 10],
    [error_0_omega, error_1_omega, error_3_omega, error_5_omega, error_10_omega],
    "o--",
    c="#304f9e",
    label=r"$\omega_0$",
)
plt.xlabel("nosie(%)")
plt.ylabel("relative error(%)")
plt.legend(
    # loc="upper right",
    frameon=False,
    fontsize="small",
)


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
# subfigs[3,0].text(0, 1, 'G', fontsize=18, style='normal', ha='left',va='top', wrap=True)
# subfigs[3,1].text(0, 1, 'H', fontsize=18, style='normal', ha='left',va='top', wrap=True)


# plt.savefig(r'大图反问题单孤子')
plt.show()
