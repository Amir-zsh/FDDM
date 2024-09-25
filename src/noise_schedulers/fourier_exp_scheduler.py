from typing import List, Optional, Tuple, Union
import torch
import math
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from noise_schedulers.fourier_scheduler import FourierScheduler


def extract_inefficient(a, t, x_shape):
    batch_size = x_shape[0]
    channnels = x_shape[1]
    out = torch.zeros(x_shape, dtype=torch.float32)
    for t_in in range(batch_size):
        out[t_in][:][:][:] = a[t[t_in]][:][:].unsqueeze(0).repeat(channnels, 1, 1)
    return out


# Extract the noise matrix for each sample in the batch
def extract(a, t, x_shape):
    channels = x_shape[1]
    assert channels == 1 or channels == 3

    # Select the noise matrix for each sample in the batch
    out = a[t]

    # Another way to do the above operation
    # t = t.unsqueeze(1).unsqueeze(2)
    # t = t.expand(-1, *x_shape[2:])
    # out = a.gather(0, t)

    # Repeat the sinlge channel noise to 3 channels
    out = out.unsqueeze(1).repeat(1, channels, 1, 1)
    assert out.shape == x_shape, f"Output shape: {out.shape}, Input Shape: {x_shape} "
    return out


# Secant hyperbolic function
def sech(x):
    return 1.0 / torch.cosh(x)


def cosine_energy_level(k1, k2):
    return -torch.cos(torch.tensor(math.pi) * k1) - torch.cos(
        torch.tensor(math.pi) * k2
    )


def fermi_modifier(x, m):
    assert (x >= 0).all() and (x <= 1).all()
    t = torch.exp(x * m) - 1
    return t / torch.max(t)


class FourierExpScheduler(FourierScheduler):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int,
        temperature: float = 0.1,
        fermi_start: float = 2.25,
        fermi_end: float = -2.25,
        image_size: int = 32,
        device: Union[str, torch.device] = None,
        fermi_muliplier=5,
    ):
        self.temperature = temperature
        self.precision = image_size
        self.mu = torch.linspace(fermi_start, fermi_end, num_train_timesteps)
        fermi_length = fermi_start - fermi_end
        self.mu = (
            fermi_modifier((self.mu - fermi_end) / fermi_length, fermi_muliplier)
        ) * fermi_length + fermi_end

        self.num_train_timesteps = num_train_timesteps
        self.device = device

        k1_vals = torch.linspace(0, 1, self.precision)
        k2_vals = torch.linspace(0, 1, self.precision)

        k1_vals, k2_vals = torch.meshgrid(k1_vals, k2_vals)
        self.epsilone = cosine_energy_level(k1_vals, k2_vals)
        self.epsilone = self.epsilone.unsqueeze(0)
        epsilone_view = self.epsilone.expand(num_train_timesteps, -1, -1)
        mu_view = self.mu.view(-1, 1, 1)
        self.epsilone_minus_mu = epsilone_view - mu_view

        self.exp_part = torch.exp((self.epsilone_minus_mu) / temperature)
        self.alpha_bar = 1 / (torch.exp((self.epsilone_minus_mu) / temperature) + 1)

        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (0, 0, 0, 0, 1, 0), value=1.0)

        self.delta_mu = self.mu[1:] - self.mu[:-1]
        self.delta_mu = torch.cat((self.delta_mu[0:1], self.delta_mu))
        self.delta_mu = self.delta_mu.view(num_train_timesteps, 1, 1)

        one_minus_exp_delta_mu_over_t = 1 - torch.exp(self.delta_mu / self.temperature)

        self.betas = (1 - self.alpha_bar) * one_minus_exp_delta_mu_over_t
        self.betas_over_sqrt_one_minus_alpha_bar = (
            torch.sqrt(1 - self.alpha_bar) * one_minus_exp_delta_mu_over_t
        )
        self.one_minus_betas = 1.0 - self.betas
        self.sqrt_recip_one_minus_betas = torch.sqrt(1.0 / self.one_minus_betas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            1 - self.alpha_bar_prev
        ) * one_minus_exp_delta_mu_over_t

        # Moving attributes to device
        # self.alpha_bar = self.alpha_bar.to(device)
        # self.posterior_variance = self.posterior_variance.to(device)
        # self.epsilone_minus_mu = self.epsilone_minus_mu.to(device)

    # for t in range(timesteps):
    #     epsilone_minus_mu[t][:][:] = epsilone - mu_t[t]
