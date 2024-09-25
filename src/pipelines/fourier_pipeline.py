from utils import idct2
from typing import List, Optional, Tuple, Union

import torch
from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from noise_schedulers.fourier_scheduler import extract


class FourierPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, denoise_algo, loss_type, ddim_sigma):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
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

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.image_size, int):
            image_shape = (
                batch_size,
                self.unet.channels,
                self.unet.image_size,
                self.unet.image_size,
            )
        else:
            image_shape = (batch_size, self.unet.channels, *self.unet.image_size)

        device = next(self.unet.parameters()).device

        if noise != None:
            image = noise
        else:
            image = randn_tensor(image_shape, generator=generator, device=device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        output_images = []
        output_pred_noises = []
        # print(self.scheduler.timesteps)
        untransformed_images = []
        model_outputs = []
        output_images.append(image)
        alpha_bar = self.scheduler.alpha_bar.to(device=device)
        exp_part = self.scheduler.exp_part.to(device=device)

        # logging.info(self.denoise_algo, self.ddim_sigma)

        for t in self.progress_bar(self.scheduler.timesteps):
            # print(t)
            # 1. predict noise model_output
            # print("---------  Input  ------------")
            # print(f"min = {torch.min(image)}, max = {torch.max(image)}, isNane: {torch.isnan(image).any()}")
            _t = torch.full((batch_size,), t, device=device)
            model_output = self.unet(image, _t)

            # 2. compute previous image: x_t -> x_t-1

            # print(f"min = {torch.min(model_output)}, max = {torch.max(model_output)}, isNane: {torch.isnan(model_output).any()}")
            # print("---------  Output  ------------")

            if self.denoise_algo == "soft_ddim":
                nu = self.ddim_sigma
                prev_t = self.scheduler.previous_timestep(t)
                prev_t = torch.full((batch_size,), prev_t, device=device)
                alpha_bar_t = extract(alpha_bar, _t, image.shape).to(device)
                alpha_bar_prev = extract(alpha_bar, prev_t, image.shape).to(device)
                exp_part_t = extract(exp_part, _t, image.shape).to(device)
                exp_part_prev = extract(exp_part, prev_t, image.shape).to(device)

                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

                alpha_bar_prev_over_current = (exp_part_t + 1) / (exp_part_prev + 1)
                sqrt_alpha_bar_prev_over_current = torch.sqrt(
                    alpha_bar_prev_over_current
                )

                one_minus_alpha_bar_current_over_prev = (exp_part_t - exp_part_prev) / (
                    exp_part_t + 1
                )
                one_minus_alpha_bar_prev_over_current = (exp_part_prev / exp_part_t) * (
                    (exp_part_t + 1) / (exp_part_prev + 1)
                )

                sigma = nu * torch.sqrt(
                    one_minus_alpha_bar_prev_over_current
                    * one_minus_alpha_bar_current_over_prev
                )

                x_prev_freq = sqrt_alpha_bar_prev_over_current * (
                    image - sqrt_one_minus_alpha_bar_t * model_output
                )

                det_noise_coeff = torch.nan_to_num(
                    torch.sqrt(1 - alpha_bar_prev - sigma**2)
                )

                det_noise = det_noise_coeff * model_output
                stoc_noise = sigma * torch.randn(image.shape).to(device)

                noise_prev = det_noise + stoc_noise

                if t != 0:
                    # x_prev_freq = x_0_freq * sqrt_alpha_bar_prev # Estimate sclaled image at t-1
                    image = x_prev_freq + noise_prev  # estimate output for t-1
                else:
                    image = x_prev_freq

            else:
                res = self.scheduler.step(
                    model_output,
                    t,
                    image,
                    generator=generator,
                    return_dict=True,
                    return_pred_noise=return_pred_noise,
                )
                image = res[0]

            if return_pred_noise:
                pred_noise = res[1]
                output_pred_noises.append(pred_noise)

            output_image = image
            model_outputs.append(output_image)

            if convert_to_image_space:
                left_pad = pad[0] or None
                right_pad = -pad[2] or None
                top_pad = pad[1] or None
                bottom_pad = -pad[3] or None
                output_image = output_image[..., left_pad:right_pad, top_pad:bottom_pad]
                if transform_from_scaled_fourier:
                    output_image = transform_from_scaled_fourier(output_image)
                else:
                    output_image = idct2(torch.sinh(output_image))

                untransformed_images.append(output_image)
                output_image = output_image / 2 + 0.5

            output_images.append(output_image)

        print(torch.min(image), torch.max(image))

        if return_pred_noise:
            return output_images, output_pred_noises

        if return_all:
            return output_images, untransformed_images

        if return_image_and_fourier:
            return output_images[-1], model_outputs[-1]

        return output_images[-1]
