import torch
from noise_schedulers.fourier_scheduler import extract
import sys
import os
import torch
from utils import dct2, idct2
import torch

sys.path.insert(0, os.getcwd())
import logging
from typing import List, Optional, Tuple, Union
from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from utils import (
    patchify,
    unpatchify,
    group_dct_components,
    ungroup_dct_components,
    ungroup_components_stat,
    get_dataset_stats,
)


class FDDMPipeline(DiffusionPipeline):
    def __init__(
        self,
        process_config,
        unet,
        scheduler,
        denoise_algo,
        loss_type,
        ddim_sigma,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.process_config = process_config
        self.denoise_algo = denoise_algo
        self.ddim_sigma = ddim_sigma
        self.loss_type = loss_type

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        return_all: bool = False,
        convert_to_image_space: bool = True,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        return_pred_noise: bool = False,
        noise=None,
        pad=[0, 0, 0, 0],
        return_image_and_fourier=False,
        transform_from_scaled_fourier=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        print(f"Denoising algo: {self.denoise_algo}")
        # Sample gaussian noise to begin loop

        input_channels = self.process_config.channels
        image_size = self.process_config.image_size
        patch_size = self.process_config.patch_size

        image_shape = (
            batch_size,
            input_channels * (image_size // patch_size) ** 2,
            patch_size,
            patch_size,
        )

        device = next(self.unet.parameters()).device

        dataset_mean, dataset_std = get_dataset_stats(
            dataset=self.process_config.dataset,
            patch_size=patch_size,
            dummy=not self.process_config.norm_freq_comps,
        )

        reshaped_std = ungroup_components_stat(
            dataset_std,
            input_channels=input_channels,
            patch_size=patch_size,
            num_patches_per_dim=image_size // patch_size,
        ).to(device=device)

        reshaped_mean = ungroup_components_stat(
            dataset_mean,
            input_channels=input_channels,
            patch_size=patch_size,
            num_patches_per_dim=image_size // patch_size,
        ).to(device=device)

        start_noise = randn_tensor(image_shape, generator=generator, device=device)
        image = start_noise
        image = patchify(image, patch_size=patch_size)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        alpha_bar = self.scheduler.alpha_bar.to(device=device)
        exp_part = self.scheduler.exp_part.to(device=device)
        output_images = []
        model_outputs = []
        logging.info(f"loss_type: {self.loss_type}")
        logging.info(f"DDIM Sigma: {self.ddim_sigma}")
        logging.info("Version: 5.0")
        self.unet.eval()
        logging.info(
            f"inference_steps; num: {len(self.scheduler.timesteps)}, min: {torch.min(self.scheduler.timesteps)},max: {torch.max(self.scheduler.timesteps)}"
        )
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            _t = torch.full((batch_size,), t, device=device)
            reshaped_image = group_dct_components(
                image,
                patch_size=patch_size,
                num_patches_per_dim=image_size // patch_size,
                num_channels=input_channels,
            )
            model_out = self.unet(reshaped_image, _t)

            # noise_t = torch.randn(x_0.shape).to(device)
            if self.loss_type == "full_img":
                x_0 = model_out
                x_0 = x_0.clamp(-1, 1)
                pach_x_0 = patchify(model_out, patch_size)
                patch_x_0_freq = dct2(pach_x_0)
                patch_x_0_freq = (pach_x_0 - reshaped_mean) / reshaped_std
                # reshaped_x_0_freq = group_dct_components(patch_x_0_freq, patch_size=patch_size, num_patches_per_dim=image_size // patch_size, num_channels=input_channels)
            elif (
                self.loss_type == "patch_image"
                or self.loss_type == "reshaped_freq"
                or self.loss_type == "reshaped_freq_full_img"
                or self.loss_type == "reshaped_freq_full_img_v2"
            ):
                reshaped_x_0_freq = model_out
                patch_x_0_freq = ungroup_dct_components(
                    reshaped_x_0_freq,
                    patch_size=patch_size,
                    num_patches_per_dim=image_size // patch_size,
                )
                if self.process_config.clip_sample:
                    patch_x_0_freq = self.clamp_dct(
                        patch_x_0_freq,
                        reshaped_mean,
                        reshaped_std,
                        unnormalize=(self.loss_type == "reshaped_freq_full_img"),
                    )
            elif self.loss_type == "reshaped_freq_v2":
                reshaped_x_0_freq = model_out
                patch_x_0_freq = ungroup_dct_components(
                    reshaped_x_0_freq,
                    patch_size=patch_size,
                    num_patches_per_dim=image_size // patch_size,
                )
                if self.process_config.clip_sample:
                    patch_x_0_freq = self.clamp_dct(
                        patch_x_0_freq, reshaped_mean, reshaped_std
                    )
            else:
                raise Exception(
                    "loss_type must be original_img, noise_img, original_freq, noise_freq"
                )

            if t != 0:
                # DDIM https://arxiv.org/abs/2010.02502
                # Soft DDIM https://openreview.net/pdf?id=W98rebBxlQ. appendix B
                x_0_freq = patch_x_0_freq
                __prev_t = self.scheduler.previous_timestep(t)
                _prev_t = torch.full((batch_size,), __prev_t, device=device)
                if self.denoise_algo == "ddim" or self.denoise_algo == "soft_ddim":
                    if self.denoise_algo == "ddim":
                        nu = 0
                    else:
                        nu = self.ddim_sigma
                    sqrt_alpha_bar_t = extract(
                        torch.sqrt(alpha_bar), _t, x_0_freq.shape
                    ).to(device)
                    sqrt_alpha_bar_prev = extract(
                        torch.sqrt(alpha_bar), _prev_t, x_0_freq.shape
                    ).to(device)
                    exp_part_t = extract(exp_part, _t, x_0_freq.shape).to(device)
                    exp_part_prev = extract(exp_part, _prev_t, x_0_freq.shape).to(
                        device
                    )

                    # (1 - alpha_t-1) /(1 - alpha_t)  //Appendinx B.3 of DDIM paper
                    one_minus_alpha_bar_prev_over_current = (
                        exp_part_prev / exp_part_t
                    ) * ((exp_part_t + 1) / (exp_part_prev + 1))

                    # 1 - (alpha_t /  alpha_t-1) //Appendinx B.3 of DDIM paper
                    one_minus_alpha_bar_current_over_prev = (
                        exp_part_t - exp_part_prev
                    ) / (exp_part_t + 1)

                    sigma = nu * torch.sqrt(
                        one_minus_alpha_bar_prev_over_current
                        * one_minus_alpha_bar_current_over_prev
                    )

                    x_t_freq = image
                    y_t_freq = (
                        x_0_freq * sqrt_alpha_bar_t
                    )  # Coarse predition of scaled image at step t
                    noise_hat = x_t_freq - y_t_freq  # estimate noise at scaled t

                    # sqrt( (1 - alpha_bar_t-1 - sigma(nu)) / (1 - alpha_bar_t)) ) //Appendinx B.2 of DDIM paper
                    det_noise_coeff = torch.sqrt(
                        (((1 - sigma**2) * exp_part_prev - sigma**2) / exp_part_t)
                        * ((exp_part_t + 1) / (exp_part_prev + 1))
                    )

                    # det_noise_coeff1 = torch.sqrt(
                    #     one_minus_alpha_bar_prev_over_current * (1 - nu**2 * (1 - (exp_part_prev / exp_part_t)))
                    # )
                    # assert torch.allclose(det_noise_coeff1, det_noise_coeff, atol=1e-5)

                    det_noise = noise_hat * det_noise_coeff
                    stoc_noise = sigma * torch.randn(x_0_freq.shape).to(device)

                    noise_prev = det_noise + stoc_noise

                    x_prev_freq = (
                        x_0_freq * sqrt_alpha_bar_prev
                    )  # Estimate sclaled image at t-1
                    image = x_prev_freq + noise_prev  # estimate output for t-1
                else:
                    raise Exception("denoise_algo must be 1 or 2 or 3")

            else:
                image = patch_x_0_freq
            # 2. compute previous image: x_t -> x_t-1
            # image = image - pred_images_t + pred_images_prev

            output_image = image
            output_image = output_image * reshaped_std + reshaped_mean

            model_outputs.append(
                unpatchify(output_image, num_patches_per_dim=image_size // patch_size)
            )

            output_image = idct2(output_image)

            output_image = unpatchify(
                output_image, num_patches_per_dim=image_size // patch_size
            )

            # model_outputs.append(tpt)

            if convert_to_image_space:
                left_pad = pad[0] or None
                right_pad = -pad[2] or None
                top_pad = pad[1] or None
                bottom_pad = -pad[3] or None
                output_image = output_image[..., left_pad:right_pad, top_pad:bottom_pad]
                if transform_from_scaled_fourier:
                    output_image = transform_from_scaled_fourier(output_image)

            # output_images.append(x_0 / 2 + 0.5)
            output_images.append(output_image)

        print(torch.min(output_image), torch.max(output_image))

        self.unet.train()

        if return_all:
            return output_images, model_outputs

        if return_image_and_fourier:
            return output_images[-1], model_outputs[-1]

    def dct_to_image_patch(self, dct_patches, mean, std, unnormalize=True):
        if unnormalize:
            dct_patches = dct_patches * std + mean
        _out = idct2(dct_patches)
        return _out

    def image_to_dct_patch(self, image_patches, mean, std, normalize=True):
        _out = dct2(image_patches)
        if normalize:
            _out = (_out - mean) / std
        return _out

    def clamp_dct(self, dct_patches, mean, std, unnormalize=True):
        _out = self.dct_to_image_patch(dct_patches, mean, std, unnormalize)
        _out = _out.clamp(-1, 1)
        _out = self.image_to_dct_patch(_out, mean, std, normalize=True)
        return _out
