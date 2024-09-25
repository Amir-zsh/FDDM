# %%
"""
Fourier process combined with the techniques from the paper "Learning in Frequency Domain": https://arxiv.org/abs/2002.12416
"""

import time
from torchvision.transforms import (
    Compose,
    ToTensor,
    CenterCrop,
    Resize,
    RandomHorizontalFlip,
)
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import sys
import os
import torch
from utils import dct2, idct2
import torch

sys.path.insert(0, os.getcwd())
from noise_schedulers import fourier_scheduler
import logging
from custom_datasets import DatasetFromDir
from noise_schedulers import fourier_exp_scheduler
from fourier_process import FourierRunner, get_parser, parse_config, simple_ema
from pipelines.fddm_pipeline import FDDMPipeline
from models.ddpm_unet_LiF import ddpm_unet_LiF
from models.DiT_LiF import DiT_LiF_models
import kornia
from utils import (
    patchify,
    unpatchify,
    group_dct_components,
    ungroup_dct_components,
    ungroup_components_stat,
    get_dataset_stats,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FDDMRunner(FourierRunner):
    def transform_to_scaled_fourier(self, image):
        assert (
            image.shape[-3] == 1 or image.shape[-3] == 3
        ), "Image must be grayscale or RGB"
        if image.shape[-3] == 3:
            image = kornia.color.rgb_to_ycbcr(image)
        image = image * 2 - 1
        return image

    def transform_from_scaled_fourier(self, scaled_image_batch):
        assert (
            scaled_image_batch.shape[-3] == 1 or scaled_image_batch.shape[-3] == 3
        ), "Image must be grayscale or YCbCr"
        scaled_image_batch = (scaled_image_batch + 1) / 2
        if scaled_image_batch.shape[-3] == 3:
            scaled_image_batch = kornia.color.ycbcr_to_rgb(scaled_image_batch)
        scaled_image_batch = scaled_image_batch.clamp(0, 1)

        return scaled_image_batch

    def get_trainset(self):
        transform_train = Compose(
            [
                Resize(self.config.image_size),
                RandomHorizontalFlip(),
                CenterCrop(self.config.image_size),
                ToTensor(),
            ]
        )

        if self.config.dataset == "CelebA":
            trainset = DatasetFromDir(
                "../data/img_align_celeba",
                self.config.image_size,
                transform=transform_train,
            )

        elif self.config.dataset == "CIFAR10":
            trainset = datasets.CIFAR10(
                root="../data",
                train=True,
                download=True,
                transform=transform_train,
            )
        elif self.config.dataset == "FashionMNIST":
            trainset = datasets.FashionMNIST(
                root="../data",
                train=True,
                download=True,
                transform=transform_train,
            )
        else:
            raise ValueError(
                "Dataset not supported. Please choose from CelebA, CIFAR10, FashionMNIST"
            )

        return trainset

    def get_model(self):
        img_size = self.config.image_size // self.config.patch_size
        ch_32_mul = (
            self.config.channels * 8**2
        )  # Number of channels compatible with 32 groupnorm
        if self.config.patch_size == self.config.image_size:
            dct_channels = self.config.channels
        else:
            dct_channels = self.config.channels * self.config.patch_size**2

        keep_dim = not (
            self.config.patch_size == 1
            or self.config.patch_size == self.config.image_size
        )
        if self.config.unet == "ddpm_unet_LiF":
            logging.info(f"Using ddpm_unet_LiF")
            model = ddpm_unet_LiF(
                resolution=self.config.image_size,
                T=self.config.num_train_timesteps,
                in_channels=dct_channels,
                ch=max([192, dct_channels]),
                # ch=192,
                ch_mult=[1, 2, 2, 2],
                attn=[1],
                num_res_blocks=2,
                keep_dim=keep_dim,
                process_config=self.config,
                dropout=0.1,
            )
        elif self.config.unet == "DiT_LiF":
            model = DiT_LiF_models[f"{self.config.DiT_model}/{self.config.patch_size}"](
                input_size=self.config.image_size,
                in_channels=self.config.channels,
                num_classes=self.config.image_size,
                process_config=self.config,
            )
        else:
            print(self.config.unet)
            raise ValueError("Unet type not recognized")
        logging.info(f"number of parameters: {count_parameters(model)}")
        return model

    def get_pipeline(self, model, scheduler):
        pipeline = FDDMPipeline(
            process_config=self.config,
            unet=model,
            scheduler=scheduler,
            denoise_algo=self.config.denoise_algo,
            loss_type=self.config.loss_type,
            ddim_sigma=self.config.ddim_sigma,
        )
        return pipeline

    @torch.no_grad()
    def get_noisy_images(self, clean_images, noise, timesteps):
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        return noisy_images

    def sample(self, num, ema=False, seed=0, return_all_steps=False):
        if ema:
            pipeline = self.pipeline_ema
        else:
            pipeline = self.pipeline
        images, fourier = pipeline(
            batch_size=num,
            generator=torch.manual_seed(seed),
            num_inference_steps=self.config.num_inference_steps,
            pad=self.config.pad,
            return_image_and_fourier=True,
            transform_from_scaled_fourier=self.transform_from_scaled_fourier,
            return_all=return_all_steps,
        )

        return images, fourier

    def get_loss(
        self,
        timesteps,
        clean_images,
        reshaped_norm_clean_images_freq,
        model_out,
    ):
        # The model outputs patchified output which is compared to patchified frequency
        # Model channels 192 -> 3
        # The model has to learn iDCT
        if self.config.loss_type == "full_img":
            loss = F.mse_loss(clean_images, model_out)

        # The model outputs patchified frequcny which is compared to patchified frequency
        # Model channels 192 -> 192
        # the model does not need to learn the iDCT
        elif self.config.loss_type == "reshaped_freq":
            loss = F.mse_loss(
                reshaped_norm_clean_images_freq, model_out, reduction="none"
            )

        # The model outputs patchified frequcny which is compared to full image
        # Model channels 192 -> 192
        # Does not need to learn iDCT
        elif (
            self.config.loss_type == "reshaped_freq_full_img"
            or self.config.loss_type == "reshaped_freq_full_img_v2"
        ):
            patch_size = self.config.patch_size
            image_size = self.config.image_size
            input_channels = self.config.channels
            out_patch_dct = ungroup_dct_components(
                model_out,
                patch_size=patch_size,
                num_patches_per_dim=image_size // patch_size,
            )
            # out_patch_dct = (
            #     out_patch_dct - self.reshaped_mean
            # ) / self.reshaped_std

            # Unnormalize
            if self.config.loss_type == "reshaped_freq_full_img":
                out_patch_dct = out_patch_dct * self.reshaped_std + self.reshaped_mean
            out_patch_image = idct2(out_patch_dct)
            out_image = unpatchify(
                out_patch_image, num_patches_per_dim=image_size // patch_size
            )
            loss = F.mse_loss(clean_images, out_image, reduction="none")

        else:
            raise Exception(
                "loss_type must be full_img, patch_img, reshaped_freq, reshaped_freq_full_img"
            )

        return loss

    def get_scheduler(self):
        if self.config.fermi_schedule == "linear":
            logging.info("Using linear fermi schedule")
            noise_scheduler = fourier_scheduler.FourierScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.patch_size,
            )
        elif self.config.fermi_schedule == "exp":
            logging.info("Using exp fermi schedule")
            noise_scheduler = fourier_exp_scheduler.FourierExpScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.patch_size,
                fermi_muliplier=self.config.fermi_muliplier,
            )

        return noise_scheduler

    def run(self):
        _step = self.init_step
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        model = self.model
        device = self.config.device
        # ema = self.ema
        ema_model = self.ema_model
        patch_size = self.config.patch_size
        image_size = self.config.image_size
        input_channels = self.config.channels
        patch_num_per_channel = (image_size // patch_size) ** 2
        patchified_input_shape = (
            patch_num_per_channel * input_channels,
            patch_size,
            patch_size,
        )

        single_batch_iterations = 1000000 if self.config.overfit else 1

        dataset_mean, dataset_std = get_dataset_stats(
            dataset=self.config.dataset,
            patch_size=self.config.patch_size,
            dummy=not self.config.norm_freq_comps,
        )

        if self.config.norm_freq_comps:
            logging.info("normalizing frequency components")
        else:
            logging.info("not normalizing frequency components")

        reshaped_mean = ungroup_components_stat(
            dataset_mean,
            input_channels=input_channels,
            patch_size=patch_size,
            num_patches_per_dim=image_size // patch_size,
        ).to(device)

        reshaped_std = ungroup_components_stat(
            dataset_std,
            input_channels=input_channels,
            patch_size=patch_size,
            num_patches_per_dim=image_size // patch_size,
        ).to(device)

        self.reshaped_std = reshaped_std
        self.reshaped_mean = reshaped_mean
        input_sum = 0
        input_num = 0
        input_squared_sum = 0
        only_input_stat = True
        for epoch in range(self.config.epochs):
            logging.info(f"epoch: {epoch}")
            for step, data in enumerate(self.dataloader, 0):
                for l in range(single_batch_iterations):
                    _step += 1

                    # get the inputs; data is a list of [inputs, labels]
                    # clean_images, labels = data[0].to(device), data[1].to(device)
                    clean_images = data[0]

                    patch_clean_images = patchify(clean_images, patch_size=patch_size)

                    batch_size = clean_images.shape[0]

                    assert patch_clean_images.shape[1:] == patchified_input_shape

                    # # Algorithm 1 line 3: sample t uniformally for every example in the batch (algo is in the blogpost)
                    timesteps = self.get_train_timesteps(
                        0, self.get_num_train_timesteps(), batch_size, device
                    )

                    noise = self.get_noise(
                        patch_clean_images.shape, patch_clean_images.device
                    )

                    patch_clean_images_freq = dct2(patch_clean_images)

                    # Normalize the images in the frequency domain
                    norm_patch_clean_images_freq = (
                        patch_clean_images_freq - reshaped_mean
                    ) / reshaped_std

                    # assert(reshaped_clean_images_freq.shape == reshaped_noise.shape)

                    # Add noise to the clean images according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)

                    norm_patch_noisy_images_freq = self.get_noisy_images(
                        norm_patch_clean_images_freq, noise, timesteps
                    )

                    reshaped_noisy_images_freq = group_dct_components(
                        norm_patch_noisy_images_freq,
                        patch_size=patch_size,
                        num_patches_per_dim=image_size // patch_size,
                        num_channels=input_channels,
                    )

                    reshaped_norm_clean_images_freq = group_dct_components(
                        norm_patch_clean_images_freq,
                        patch_size=patch_size,
                        num_patches_per_dim=image_size // patch_size,
                        num_channels=input_channels,
                    )

                    if _step % 100 == 0:
                        if only_input_stat:
                            pass
                    t = time.time()
                    model_out = model(reshaped_noisy_images_freq, timesteps)

                    error = self.get_loss(
                        timesteps,
                        clean_images,
                        reshaped_norm_clean_images_freq,
                        model_out,
                    )
                    loss = error.mean()
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.modify_grads(model)

                    optimizer.step()
                    lr_scheduler.step()
                    t2 = time.time()
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        simple_ema(unwrapped_model, ema_model, self.config.ema_decay)
                    if self.accelerator.is_main_process:
                        if _step % 100 == 0:
                            logging.info(f"Step time: {t2 - t}")
                            logging.info(f"Step: {_step}, Loss: {loss.item()}")
                            if _step % 2000 == 0:
                                t_error = error.mean(dim=(1, 2, 3))
                                print(t_error.shape)
                                print(t_error)

                                self.writer.add_histogram(
                                    "Loss/timestep", t_error, _step
                                )
                            self.writer.add_scalar("Loss/train", loss, _step)
                            self.writer.add_scalar(
                                "Learning Rate",
                                lr_scheduler.get_last_lr()[0],
                                _step,
                            )

                        self.sample_and_save(
                            epoch,
                            _step,
                            unpatchify(
                                norm_patch_clean_images_freq,
                                self.config.image_size // self.config.patch_size,
                            ),
                        )

                    # zero the parameter gradient
                    optimizer.zero_grad()

                if _step == self.config.max_steps:
                    break
            if _step == self.config.max_steps:
                break

        self.save_state("final")


if __name__ == "__main__":
    parser = get_parser()
    config = parse_config(parser)
    runner = FDDMRunner(config)
    runner.run()
