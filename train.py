import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


DEVICE        = torch.device("cuda" if torch.has_cuda else "cpu")
INPUT_DIM     = 784
HIDDEN_DIM    = 200
Z_DIM         = 20
NUM_EPOCHS    = 10
BATCH_SIZE    = 32
LEARNING_RATE = 1e-8

dataset = datasets.MNIST(
    root="dataset/", 
    download=True, 
    train=True, 
    transform=transforms.ToTensor()
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(input_dim=INPUT_DIM, h_dim=HIDDEN_DIM, z_dim=Z_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()


for epoch in range(NUM_EPOCHS):
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    for ix, (X, _) in pbar:
        X = X.to(DEVICE).view(X.shape[0], INPUT_DIM)
        X_recon, mu, std = model(X)

        recon_loss = criterion(X_recon, X)
        kl_div = torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))

        loss = recon_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

