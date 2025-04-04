import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt

class DDPM():
    def __init__(self, n: int, net: nn.Module, input_shape):
        """
        n (int): number of timesteps
        input_shape: shape of one sample, for example (256, 256, 3) for an image
        """
        self.n = n
        self.betas = torch.linspace(0, 0.02, n+1) 
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
        self.net = net
        self.input_shape = input_shape

    def embed_time(self, X, t):
        assert len(X.shape) == 2
        t_normalized = (2*t-self.n)/self.n
        input = torch.cat((X, t_normalized), dim=1)
        return input


    def train(self, loader, n_epochs):
        epoch_loss = []
        optimizer = AdamW(self.net.parameters())
        for e in range(n_epochs):
            lossi = []
            for batch in loader:
                X = batch[0] # (batch_size, ...)
                t = torch.randint(1, self.n, (X.shape[0],1))
                noise = torch.randn_like(X) 
                X_noised = torch.sqrt(self.alpha_bars[t])*X + \
                    torch.sqrt(1-self.alpha_bars[t])*noise
                input = self.embed_time(X_noised, t)
                out = self.net(input)
                assert out.shape == noise.shape, f"{out.shape = }, {noise.shape = }"
                loss = torch.mean((out - noise)**2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossi.append(loss.item())
            print(f"epoch {e}: mean loss = {np.mean(lossi)}")
            epoch_loss.append(np.mean(lossi))
        plt.plot(epoch_loss)

    def sample(self, n_samples, eta=1):
        x = torch.randn((n_samples,) + self.input_shape)
        for t in range(self.n, 0, -1):
            noise = eta*torch.randn_like(x)
            input = self.embed_time(x, t*torch.ones(x.shape[0], 1))
            noise_pred = self.net(input) * self.betas[t]/torch.sqrt(1-self.alpha_bars[t])
            x = (x - noise_pred)/torch.sqrt(self.alphas[t]) + self.sigmas[t] * noise
        return x