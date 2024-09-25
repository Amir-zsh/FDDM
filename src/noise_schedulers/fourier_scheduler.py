from typing import List, Optional, Tuple, Union
import torch
import numpy as np
import math
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


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
    # assert channels == 1 or channels == 3

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


class FourierScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int,
        temperature: float = 0.1,
        fermi_start: float = 2.25,
        fermi_end: float = -2.25,
        image_size: int = 32,
        device: Union[str, torch.device] = None,
    ):
        self.temperature = temperature
        self.precision = image_size
        self.mu = torch.linspace(fermi_start, fermi_end, num_train_timesteps)
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
        self.one_over_alpha_bar = self.exp_part + 1
        self.alpha_bar = 1 / self.one_over_alpha_bar

        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (0, 0, 0, 0, 1, 0), value=1.0)

        self.delta_mu = self.mu[1] - self.mu[0]

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

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
        return_step_noise: bool = False,
    ) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn_like(original_samples)
        assert noise.shape == original_samples.shape
        assert original_samples.shape[0] == timesteps.shape[0]

        alpha_bar = self.alpha_bar.to(device=original_samples.device)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_bar_t = extract(
            torch.sqrt(alpha_bar), timesteps, original_samples.shape
        )
        sqrt_one_minus_alpha_bar_t = extract(
            torch.sqrt(1 - alpha_bar), timesteps, original_samples.shape
        )

        noisy_samples = (
            sqrt_alpha_bar_t * original_samples + sqrt_one_minus_alpha_bar_t * noise
        )

        if return_step_noise:
            betas = self.betas.to(device=original_samples.device)
            sqrt_betas_t = extract(torch.sqrt(betas), timesteps, original_samples.shape)
            sqrt_recip_one_minus_betas = self.sqrt_recip_one_minus_betas.to(
                device=original_samples.device
            )
            sqrt_recip_one_minus_betas_t = extract(
                sqrt_recip_one_minus_betas, timesteps, original_samples.shape
            )
            return (noisy_samples, sqrt_recip_one_minus_betas_t * sqrt_betas_t * noise)

        return noisy_samples

    def get_alpha_bar(self, timesteps: torch.IntTensor, shape):
        # return extract(torch.sqrt(self.alpha_bar), timesteps, shape)
        return extract(torch.sqrt(self.alpha_bar), timesteps, shape)
        # return extract(self.posterior_variance, timesteps, shape)
        # return extract(self.betas_over_sqrt_one_minus_alpha_bar, timesteps, shape)
        # return extract(self.sqrt_recip_one_minus_betas, timesteps, shape)

    def get_loss_wrapper(self, timesteps, shape, t2, func="sech", normalize=False):
        if func == "sech":
            epsilone_minus_mu = self.epsilone_minus_mu.to(device=timesteps.device)
            epsilone_minus_mu_t = extract(epsilone_minus_mu, timesteps, shape)
            wrapper = sech(epsilone_minus_mu_t / t2)
        elif func == "noise_coeff":
            alpha_bar = self.alpha_bar.to(device=timesteps.device)
            sqrt_one_minus_alpha_bar_t = extract(
                torch.sqrt(1 - alpha_bar), timesteps, shape
            )
            wrapper = sqrt_one_minus_alpha_bar_t
        else:
            raise "Wrong Wrapper Function"
        if normalize:
            # max_values = torch.max(wrapper.view(shape[0], -1), dim=1, keepdim=True)[0].view(-1, *(len(shape[1:]) * (1,)))
            max_values = torch.norm(
                wrapper, dim=(2, 3), keepdim=True
            )  # Normalize by norm
            wrapper = wrapper / max_values
        return wrapper

    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_pred_noise: bool = False,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        device = model_output.device
        # t = timestep
        batch_size = sample.shape[0]
        t = torch.full((batch_size,), timestep, device=device)
        prev_t = self.previous_timestep(t)

        x = sample
        m = model_output

        # betas = self.betas.to(device)
        alpha_bar = self.alpha_bar.to(device=device)
        betas_over_sqrt_one_minus_alpha_bar = (
            self.betas_over_sqrt_one_minus_alpha_bar.to(device)
        )
        sqrt_recip_one_minus_betas = self.sqrt_recip_one_minus_betas.to(device=device)
        posterior_variance = self.posterior_variance.to(device=device)

        # betas_t = extract(betas, t, x.shape).to(device)
        betas_over_sqrt_one_minus_alpha_bar_t = extract(
            betas_over_sqrt_one_minus_alpha_bar, t, x.shape
        )
        sqrt_alpha_bar_t = extract(torch.sqrt(alpha_bar), t, x.shape).to(device)
        sqrt_one_minus_alpha_bar_t = extract(torch.sqrt(1 - alpha_bar), t, x.shape).to(
            device
        )
        sqrt_recip_one_minus_betas_t = extract(sqrt_recip_one_minus_betas, t, x.shape)

        temp1 = m * betas_over_sqrt_one_minus_alpha_bar_t

        pred_prev_sample = sqrt_recip_one_minus_betas_t * (x - temp1)

        if timestep != 0:
            posterior_variance_t = extract(posterior_variance, t, x.shape).to(
                device
            )  # this is sigma_t
            # posterior_variance_t = torch.unsqueeze(posterior_variance_t, dim=1)
            if torch.isnan(posterior_variance_t).any():
                assert False
            noise = torch.randn_like(x)
            if torch.isnan(torch.sqrt(posterior_variance_t) * noise).any():
                print("2", torch.sqrt(posterior_variance_t) * noise)
                assert False
            # Algorithm 2 line 4:
            pred_prev_sample = (
                pred_prev_sample + torch.sqrt(posterior_variance_t) * noise
            ).to(device)
            if torch.isnan(pred_prev_sample).any():
                print("pred_prev_sample", pred_prev_sample.shape)
                assert False

        if return_pred_noise:
            # betas = self.betas.to(device=device)
            # sqrt_betas_t = extract(torch.sqrt(betas), t, x.shape)
            ret = sqrt_recip_one_minus_betas_t * temp1
            if timestep != 0:
                ret = ret - torch.sqrt(posterior_variance_t) * noise
            return (pred_prev_sample, ret)

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError(
                "Can only pass one of `num_inference_steps` or `custom_timesteps`."
            )

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            # Replaced with DDIM implementation
            # step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

            # From  https://github.com/huggingface/diffusers/blob/v0.22.2/src/diffusers/schedulers/scheduling_ddim.py#L321
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            self.custom_timesteps = True  # Changed

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps
                if self.num_inference_steps
                else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t
