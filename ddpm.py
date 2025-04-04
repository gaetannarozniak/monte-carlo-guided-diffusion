import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

class DDPM():
    def __init__(self, n: int, net: nn.Module, input_shape):
        """
        n (int): number of timesteps
        input_shape: shape of one sample, for example (256, 256, 3) for an image
        """
        self.n = n
        self.betas = torch.linspace(0, 0.02, n+1) 
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas)
        self.net = net
        self.input_shape = input_shape

    def train(self, loader, n_epochs):
        optimizer = AdamW(self.net.parameters)
        for e in range(n_epochs):
            for batch in loader:
                X = batch[0] # (batch_size, ...)
                t = np.random.randint(1, self.n+1)
                noise = torch.randn_like(X[0]) 
                X_noised = np.sqrt(self.alpha_bars[t])*X + \
                    np.sqrt(1-self.alpha_bars[t])*noise
                out = self.net(X_noised, t)
                assert out.shape == noise.shape
                loss = torch.mean((out - noise)**2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def sample(self):
        x = torch.randn(self.input_shape)
        for t in range(self.n, 0, -1):
            noise = torch.randn(self.input_shape)
            noise_pred = self.net(x, t) * self.betas[t]/np.sqrt(1-self.alpha_bars[t])
            x = (x - noise_pred)/np.sqrt(self.alpha[t]) + self.sigmas[t] * noise
        return x