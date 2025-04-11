from diffusers import DDPMPipeline
import torch
from torchvision import transforms
from torch.distributions import Categorical
from PIL import Image
import matplotlib.pyplot as plt
from utils import plot_image
from tqdm import tqdm
import numpy as np

model_id = "google/ddpm-ema-celebahq-256"
ddpm = DDPMPipeline.from_pretrained(model_id)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using device: ", device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image = Image.open("pictures/gaetan.jpeg").convert("RGB")
tensor_image = transform(image).unsqueeze(0).to(device)
tensor_image.shape

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

n = 128
unet = ddpm.unet.to(device)
scheduler = ddpm.scheduler
scheduler.set_timesteps(n)

x = torch.randn((1, 3, 256, 256)).to(device)
timesteps = scheduler.timesteps

def compute_entropy_from_tensor(x):
    x = x.detach().cpu()
    values, counts = torch.unique(x, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))  # small value to avoid log(0)
    return entropy.item()

def predict(particles, unet, scheduler, i):
    t = timesteps[i]
    t_next = timesteps[i+1]
    alpha_bar_t_next = scheduler.alphas_cumprod[t]
    alpha_bar_t = scheduler.alphas_cumprod[t_next]
    noise = (((1 - alpha_bar_t) / (1 - alpha_bar_t_next)) * 
                          (1 - (alpha_bar_t_next / alpha_bar_t))).sqrt()
    with torch.no_grad():
        noise_pred = unet(particles, t).sample
        mean = scheduler.step(noise_pred, t, particles).prev_sample
    return mean, noise/10
    
def log_likelihood(x, mean, std):
    return - 1/2 * torch.sum((x - mean)**2 / std, dim=(1, 2, 3))

mask = torch.ones((1, 3, 256, 256)).to(device)
mask[:, :, 128:, :] = 0
plot_image(tensor_image*mask)

y = (mask*tensor_image).to(device) # (1, 3, 256, 256)
N = 32
particles = torch.randn((N, 3, 256, 256)).to(device)

for i in tqdm(range(len(timesteps)-1)):
        mean, noise = predict(particles, unet, scheduler, i)
        particles = mean + noise*torch.randn(x.shape).to(device)
        alpha_bar = scheduler.alphas_cumprod[timesteps[i]].to(device)
        alpha_bar_up =  scheduler.alphas_cumprod[timesteps[i+1]].to(device)

        log_weights = log_likelihood(alpha_bar**0.5*y, mean*mask, noise**2+1-alpha_bar) - \
            log_likelihood(alpha_bar_up**0.5*y, particles*mask, 1-alpha_bar_up)

        I = Categorical(logits=log_weights/torch.sum(log_weights)).sample((N,))
        z = torch.randn(particles.shape).to(device)
        K = noise**2 / (noise**2+1-scheduler.alphas[timesteps[i]])
        particles_unmasked = K*alpha_bar**0.5*y + (1-K)*mean*mask + (1-alpha_bar)**.5*K**.5*z*mask
        particles_masked = mean*(1-mask) #+ noise*z*(1-mask)
        particles = particles_masked+particles_unmasked
        particles = particles[I]
        print(log_weights, I, compute_entropy_from_tensor(I))
        
        
        
for i in range(N):
    plot_image(particles[i], save_dir="outputs", filename=f"particle_{i:03}.png")

del unet, scheduler, x, particles, mean, noise, log_weights, I, z, particles_unmasked, particles_masked