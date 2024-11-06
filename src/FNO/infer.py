"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

torch.manual_seed(0)
np.random.seed(0)

from torch.utils.tensorboard import SummaryWriter
import datetime


def plo(input, truth, pred, inx):
    width = 2.8

    plt.figure(figsize=(6 * width, 5 * width), dpi=400, layout="constrained")

    plt.subplot(3, 3, 1)
    plt.imshow(truth, norm=None, cmap="jet")
    plt.title("Reference")
    plt.colorbar()

    plt.subplot(3, 3, 2)
    plt.imshow(pred, norm=None, cmap="jet")
    plt.title("Predicted")
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.imshow(abs(pred - truth), norm=None, cmap="jet")
    plt.title("Absolute error")
    plt.colorbar()
    print(f"Saving figure to {folder}/epoch{ep}_{inx}")
    plt.savefig(f"{folder}/epoch{ep}_{inx}.png", bbox_inches="tight", dpi=400)
    plt.close()


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


sub = 1
S = 64
T_in = 100
T = 100  # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = 1


ntrain = 440
ntest = 60

batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 50
iterations = epochs * (ntrain // batch_size)

modes = 16
width = 32

data_all = torch.from_numpy(sio.loadmat("NLS.mat")["uu_all"].astype("float32"))  # torch.Size([1600, 128, 100])

x_train = data_all[:ntrain, ::sub, :T_in]  # torch.Size([1400, 128, 50])前50个时间
y_train = data_all[:ntrain, ::sub, T_in : T + T_in]  # torch.Size([1400, 128, 50])后50个时间

x_test = data_all[-ntest:, ::sub, :T_in]
y_test = data_all[-ntest:, ::sub, T_in : T + T_in]

print(y_train.shape)
print(y_test.shape)


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.unsqueeze(-1)  # (ntrain,ny,nx,1)
x_test = x_test.unsqueeze(-1)  # (ntest,ny,nx,1)
nx = x_test.shape[2]
ny = x_test.shape[1]
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True
)

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f"output/infer_{now}"
writer = SummaryWriter(f"{folder}")

################################################################
# training and evaluation
################################################################
device = torch.device("cpu")
model = torch.load(
    r"/home/xsy/pythoncode/myproj/WTU-AI4S/src/FNO/bright_one_ep499.pt"
)
model.to(device)
# model = FNO2d(modes, modes, width).cuda()
# print(count_params(model))


input = x_test[0:20]
true = y_test[0:20]
# true=y_normalizer.decode(true).detach().cpu().numpy()
pred = model(input).squeeze()
pred = y_normalizer.decode(pred).detach().cpu().numpy()
input = x_normalizer.decode(input.squeeze()).detach().cpu().numpy()

x_lower, x_upper, y_lower, y_upper = -5, 5, 0, 1
width = 1.5


def huatu(ind):
    fig, axs = plt.subplots(figsize=(8 * width, 2 * width), nrows=1, ncols=3)
    pred_out=np.hstack((input[ind], pred[ind]))
    true_out=np.hstack((input[ind], true[ind]))
    # plot just the positive data and save the
    # color "mappable" object returned by ax1.imshow
    pos = axs[0].imshow(
        pred_out,
        # input[ind].squeeze(),
        cmap="rainbow",
        extent=[y_lower, y_upper,x_lower, x_upper],
        # interpolation="nearest",
        # norm=norm1,
        origin="lower",  # 这个有影响，不要注释掉
        aspect="auto",
    )
    axs[0].set_title("Reference")
    # add the colorbar using the figure's method,
    # telling which mappable we're talking about and
    # which Axes object it should be near
    fig.colorbar(
        pos,
        ax=axs[0],
        shrink=0.9,
    )

    # repeat everything above for the negative data
    # you can specify location, anchor and shrink the colorbar
    neg = axs[1].imshow(
        true_out,
        # pred[ind],
        cmap="rainbow",
        extent=[y_lower, y_upper,x_lower, x_upper],
        # interpolation="nearest",
        # norm=norm1,
        origin="lower",  # 这个有影响，不要注释掉
        aspect="auto",
    )
    axs[1].set_title("Predicted")
    fig.colorbar(
        neg,
        ax=axs[1],
        location="right",
        #  anchor=(0, 0.3),
        shrink=0.9,
    )

    # Plot both positive and negative values between +/- 1.2
    pos_neg_clipped = axs[2].imshow(
        np.abs(true_out - pred_out),
        cmap="rainbow",
        extent=[y_lower, y_upper,x_lower, x_upper],
        # interpolation="nearest",
        # norm=norm1,
        origin="lower",  # 这个有影响，不要注释掉
        aspect="auto",
    )
    axs[2].set_title("Absolute error")
    # Add minorticks on the colorbar to make it easy to read the
    # values off the colorbar.
    cbar = fig.colorbar(
        pos_neg_clipped,
        ax=axs[2],
        shrink=0.9,
        # extend="both"
    )
    # cbar.minorticks_on()

    # pos4 = axs[1, 1].imshow(
    #     np.abs(true[ind] - pred[ind]),
    #     cmap="rainbow",  # "jet",  # seismic
    #     extent=[x_lower, x_upper, y_lower, y_upper],
    #     # interpolation="nearest",
    #     # norm=norm1,
    #     origin="lower",  # 这个有影响，不要注释掉
    #     aspect="auto",
    # )
    # axs[1, 1].set_title("Absolute error")
    # # add the colorbar using the figure's method,
    # # telling which mappable we're talking about and
    # # which Axes object it should be near
    # fig.colorbar(
    #     pos4,
    #     ax=axs[1, 1],
    #     shrink=0.9,
    # )

    # fig.text(
    #     0.1,
    #     0.91,
    #     "(a)",
    #     fontsize=12,
    #     va="top",
    #     ha="left",
    # )
    # fig.text(
    #     0.55,
    #     0.91,
    #     "(b)",
    #     fontsize=12,
    #     va="top",
    #     ha="left",
    # )
    # fig.text(
    #     0.1,
    #     0.49,
    #     "(c)",
    #     fontsize=12,
    #     va="top",
    #     ha="left",
    # )

    # fig.text(
    #     0.55,
    #     0.49,
    #     "(d)",
    #     fontsize=12,
    #     va="top",
    #     ha="left",
    # )
    plt.savefig(
        f"{folder}/fig_{ind}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.savefig(
    #     f"{folder}/fig_{ind}.eps",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.close()


for i in range(20):
    huatu(i)
