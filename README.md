# monte-carlo-guided-diffusion
Implementation and experiments on Cardoso et al. (2023)

- ddpm.py (Gaëtan)
has a DDPM class that has a train() and a predict() method
the train() method trains a network to predict the noise that was added (or equivalently m)
the predict(t, x) method returns the predicted m for the noise level t and the data x

predict(particles: torch.Tensor, t: int) -> mean: torch.Tensor, noise: float

- mcgdiff.py (Auguste)
has a MCGDiff class with a train() and a generate() method

