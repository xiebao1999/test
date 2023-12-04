from torch import nn
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))
x = torch.arange(start=-10, end=10, step=0.01)
y = mish(x)
plt.plot(x.numpy(), y.numpy())
plt.title("mish")
plt.savefig("./activate_image/mish.png")
