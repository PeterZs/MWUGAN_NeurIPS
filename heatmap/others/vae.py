from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2, 32)
        self.fc21 = nn.Linear(32, 10)
        self.fc22 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 32)
        self.fc4 = nn.Linear(32, 2)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

from IPython import embed
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = torch.sum(((recon_x - x) ** 2))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


from dataset import *
dataset = Sine()
batch_size = 128

def train(epoch):
    global dataset, batch_size

    dataset.reshuffle()
    train_loss = 0
    batch_idx = 0
    while batch_idx * batch_size < dataset.n_data:
        data, _ = dataset.next_batch(batch_size, batch_size * batch_idx)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataset),
                100. * batch_idx * batch_size / len(dataset),
                loss.item() / len(dataset)))
        batch_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataset)))

def train_vae():
    model.train()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    model.eval()


import numpy.random as nr
def sample(model, n=125000):
    noise = nr.randn(n, 10)
    data = model.decode(torch.Tensor(noise))
    data = data.detach().numpy()
    return data

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    print("Plotting dataset")
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], alpha=0.01)
    plt.savefig("dataset.png")
    plt.clf()
    train_vae()

    samples = sample(model)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.02)
    plt.savefig("sample_vae.png")

    plt.clf()

    samples[samples > 15] = 15
    samples[samples < -15] = -15
    samples = np.vstack((samples, np.array([[-15, -15], [15, 15]])))
    plt.hist2d(samples[:, 0], samples[:, 1], bins=(100, 100), cmap=plt.cm.jet)
    plt.axis("off")
    plt.savefig("heatmap_vae.png", dpi=300)
