import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img2hidden = nn.Linear(in_features=input_dim, out_features=h_dim)
        self.hidden2mu  = nn.Linear(h_dim, z_dim)
        self.hidden2sigma = nn.Linear(h_dim, z_dim)

        self.z2hid = nn.Linear(z_dim, h_dim)
        self.hid2img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.img2hidden(x))
        mu, sigma = self.hidden2mu(h), self.hidden2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = F.relu(self.z2hid(z))
        return torch.sigmoid(self.hid2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        z_reparam = mu + sigma * torch.randn_like(sigma)
        x_recon   = self.decode(z_reparam)
        return x_recon, mu, sigma


if __name__ == '__main__':
    x = torch.randn(1, 784)
    vae = VAE(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape, mu.shape, sigma.shape)