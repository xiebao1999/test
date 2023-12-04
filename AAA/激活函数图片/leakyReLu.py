from torch import nn
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

func = nn.LeakyReLU(negative_slope=0.05)
x = torch.arange(start=-10, end=10, step=0.01)
y = func(x)
plt.plot(x.numpy(), y.numpy())
plt.title("LeakyReLU",fontsize=20)
plt.savefig("./activate_image/LeakyRelu.png")
