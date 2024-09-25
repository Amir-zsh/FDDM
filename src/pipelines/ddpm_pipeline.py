from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

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

        image = randn_tensor(image_shape, generator=generator, device=device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        output_images = []
        untransformed_images = []
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            _t = torch.full((batch_size,), t, device=device)
            model_output = self.unet(image, _t)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

            output_image = image

            if convert_to_image_space:
                left_pad = pad[0] or None
                right_pad = -pad[2] or None
                top_pad = pad[1] or None
                bottom_pad = -pad[3] or None
                output_image = output_image[..., left_pad:right_pad, top_pad:bottom_pad]
                if transform_from_scaled_fourier:
                    output_image = transform_from_scaled_fourier(output_image)
                untransformed_images.append(output_image)
                output_image = output_image / 2 + 0.5

            output_images.append(output_image)

        print(torch.min(image), torch.max(image))

        if return_all:
            return output_images, untransformed_images

        if return_image_and_fourier:
            return output_images[-1], image

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image,)

        # return ImagePipelineOutput(images=image)
