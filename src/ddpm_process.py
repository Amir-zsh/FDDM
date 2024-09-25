import torch.nn.functional as F

import sys
import os
import torch
import torch

from diffusers import DDPMScheduler as RegularDDPMScheduler


sys.path.insert(0, os.getcwd())
from pipelines import ddpm_pipeline
import logging
from fourier_process import FourierRunner, get_parser, parse_config


def exists(x):
    return x is not None


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class DDPMRunner(FourierRunner):

    def transform_to_scaled_fourier(self, image):
        return image

    def transform_from_scaled_fourier(self, scaled_image):
        return scaled_image

    def get_scheduler(self):
        if self.config.fermi_schedule == "standard":
            logging.info("Using standard scheduler")
            noise_scheduler = RegularDDPMScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                variance_type="fixed_small_log",
            )
        else:
            raise ValueError("Scheduler not recognized")

        return noise_scheduler

    def get_pipeline(self, model, scheduler):
        pipeline = ddpm_pipeline.DDPMPipeline(unet=model, scheduler=scheduler)
        return pipeline

    def get_loss(self, timesteps, noise, noise_pred):
        loss = F.mse_loss(noise_pred, noise)
        return loss


if __name__ == "__main__":
    parser = get_parser()
    config = parse_config(parser)
    runner = DDPMRunner(config)
    runner.run()
