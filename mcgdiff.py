import torch
from torch.distributions import Categorical
from ddpm import DDPM

def log_likelihood(x, mean, std):
    return - 1/2 * torch.sum((x - mean)**2 / std, dim=1)

def mcgdiff(ddpm: DDPM, 
    y: torch.Tensor,
    dx: int,
    N: int,
    n: int,
) -> torch.Tensor:
    """
    Perform the MCGDiff algorithm in the noiseless case.
    Args:
        ddpm (DDPM): The DDPM model.
        y (torch.Tensor): the incomplete observation.
        dx (int): Dimension of x. 
        N (int): The number of samples to generate.
        n (int): The number of diffusion steps.
    Returns:
        torch.Tensor: The generated samples.
    """
    alpha_bars = ddpm.alpha_bars
    dy = len(y)
    particles = torch.zeros((n + 1, N, dx))
    particles[-1, :, :] = torch.randn((N, dx))

    for i in range(n - 1, 0, -1):
        mean, noise = ddpm.predict(particles[i + 1], i)
        log_weights = log_likelihood(alpha_bars[i]**.5 * y, mean[:,:dy], noise**2 + 1 - alpha_bars[i]) - log_likelihood(alpha_bars[i + 1]**.5 * y, particles[i + 1][:,:dy], 1 - alpha_bars[i + 1])

        I = Categorical(logits=log_weights).sample((N,))
        z_y = torch.randn((N, dy))
        z_x = torch.randn((N, dx - dy))
        K = noise ** 2 / (noise ** 2 + 1 - alpha_bars[i])
        particles[i, :, :dy] = K * alpha_bars[i]**.5 * y + (1 - K) * mean[:,:dy] + (1 - alpha_bars[i])**.5 * K**.5 * z_y
        particles[i, :, dy:] = mean[:,dy:] + noise * z_x
        particles[i] = particles[i][I]

    return particles
    