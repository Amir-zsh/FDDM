from pprint import pformat
import time
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    ToPILImage,
    CenterCrop,
    Grayscale,
    Resize,
    RandomHorizontalFlip,
    RandomCrop,
    Normalize,
    Pad,
)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pathlib import Path
import math
import uuid
import json

from torch.utils.tensorboard import SummaryWriter
import copy
import sys
import os
import torch
import argparse
from utils import dct2, idct2
import torch
import torchvision
from diffusers.optimization import get_cosine_schedule_with_warmup


sys.path.insert(0, os.getcwd())
from noise_schedulers import fourier_scheduler
from pipelines import fourier_pipeline
from models.ddpm_unet import UNet as ddpm_unetUnet
from models.DiT import DiT_models
import logging
from custom_datasets import DatasetWithCache, DatasetFromDir
from config import Config
from noise_schedulers import fourier_exp_scheduler
from accelerate import Accelerator


# Source: https://github.com/ddpm_unet/pytorch-ddpm/blob/master/main.py
@torch.no_grad()
def simple_ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
        # target_key = key[7:] if key.startswith("module") else key
        # target_dict[target_key].data.copy_(
        #     target_dict[target_key].data * decay +
        #     source_dict[key].data * (1 - decay))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FourierRunner:

    def set_config(self, config):
        config.device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
        config.image_size = config.image_size + config.pad[0] + config.pad[2]

        config.unique_id = str(uuid.uuid1())[:8]
        config.unique_name = f"{config.run_name}_{config.unique_id}"
        # Setting results folder
        config.results_folder = f"../results/{config.unique_name}"
        config.tensorboard_folder = f"../runs"
        config.results_folder = Path(config.results_folder)
        config.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else config.num_train_timesteps
        )

        self.config = config

    def __init__(self, config) -> None:
        self.accelerator = Accelerator()
        self.set_config(config)
        process_idx = self.accelerator.process_index
        if self.accelerator.is_main_process:
            self.config.results_folder.mkdir(exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format=f"{process_idx}-%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(
                        "{0}/{1}.log".format(self.config.results_folder, "loss")
                    ),
                    logging.StreamHandler(),
                ],
            )
            self.writer = SummaryWriter(
                comment=self.config.unique_name,
                log_dir=f"{self.config.tensorboard_folder}/{self.config.unique_name}",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=f"{process_idx}-%(asctime)s [%(levelname)s] %(message)s",
            )

        # log the info
        if self.accelerator.is_main_process:
            logging.info(pformat(sys.argv))
            logging.info(config)
            print(vars(config))
            with open(f"{self.config.results_folder}/config.json", "w") as f:
                serialized_config = {
                    **vars(config),
                    "results_folder": str(config.results_folder),
                    "runner": sys.argv[0],
                }
                json.dump(serialized_config, f, indent=4, sort_keys=True)

        self.init_step = 0

        self.config.device = self.accelerator.device

        self.model = self.get_model()
        self.model = self.model.to(self.config.device)

        # self.ema = EMA(
        #     self.model,
        #     beta = self.config.ema_decay,
        #     update_every = self.config.ema_update_every
        # )

        # self.ema = self.ema.to(self.config.device)
        # self.ema.ema_model.eval()
        self.ema_model = copy.deepcopy(self.model)

        self.ema_model = self.ema_model.to(self.config.device)
        self.ema_model.eval()
        self.optimizer = self.get_optimizer()

        # Load state from checkpoint

        self.scheduler = self.get_scheduler()
        self.dataloader = self.get_dataloader()

        self.pipeline = self.get_pipeline(self.model, self.scheduler)
        # self.pipeline_ema = self.get_pipeline(self.ema.ema_model, self.scheduler)
        self.pipeline_ema = self.get_pipeline(self.ema_model, self.scheduler)
        self.lr_scheduler = self.get_lr_scheduler()

        self.model, self.optimizer, self.dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader, self.lr_scheduler
            )
        )

        if self.config.checkpoint_path is not None:
            self.load_state()

    def get_lr_scheduler(self):
        if self.config.lr_scheduler == "identity":
            # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: min(step, self.config.lr_warmup_steps)
                / self.config.lr_warmup_steps,
            )
        elif self.config.lr_scheduler == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=self.config.max_steps,
            )
        else:
            raise ValueError(
                f"Unknown lr scheduler: {self.config.lr_scheduler}. Please choose from [identity, cosine]"
            )
        return lr_scheduler

    def get_dataset(self, trainset, transform):
        # dct_data = DCTScaledDataset(trainset)
        return DatasetWithCache(trainset, transform=transform)

    def get_trainset(self):
        transform_train = Compose(
            [
                Resize(self.config.image_size),
                RandomHorizontalFlip(),
                CenterCrop(self.config.image_size),
                ToTensor(),
                Lambda(lambda t: (t * 2) - 1),
            ]
        )

        if self.config.dataset == "CelebA":
            trainset = DatasetFromDir(
                "../data/img_align_celeba",
                self.config.image_size,
                transform=transform_train,
            )
            # assert self.config.image_size == 64

        elif self.config.dataset == "ImageNette":
            trainset = DatasetFromDir(
                "../data/ImageNette", self.config.image_size, transform=transform_train
            )
            # assert self.config.image_size == 64

        elif self.config.dataset == "CIFAR10_64":
            trainset = DatasetFromDir(
                "../data/CIFAR10-64x64",
                self.config.image_size,
                transform=transform_train,
            )
            # assert self.config.image_size == 64

        elif self.config.dataset == "CIFAR10":
            trainset = datasets.CIFAR10(
                root="../data", train=True, download=True, transform=transform_train
            )
            # assert self.config.image_size == 32

        elif self.config.dataset == "FashionMNIST":
            trainset = datasets.FashionMNIST(
                root="../data", train=True, download=True, transform=transform_train
            )
            # assert self.config.image_size == 28

        else:
            raise ValueError(
                "Dataset not supported. Please choose from CelebA, CIFAR10, FashionMNIST"
            )

        return trainset

    # def get_reverse_transform(self):
    #     reverse_transform = Compose([
    #         Lambda(lambda t: (t + 1) / 2),
    #         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #         Lambda(lambda t: t * 255.),
    #         Lambda(lambda t: t.numpy().astype(np.uint8)),
    #         ToPILImage(),
    #     ])
    #     return reverse_transform

    def transform_to_scaled_fourier(self, input):
        dct_image = dct2(input)
        if self.config.input_normalization == "asinh":
            scaled_output = torch.arcsinh(dct_image)
        elif self.config.input_normalization == "no_normalization":
            scaled_output = dct_image
        else:
            raise Exception("Invalid input norm")

        scaled_output = scaled_output / self.config.input_normalization_scale

        return scaled_output

    def transform_from_scaled_fourier(self, input):
        scaled_output = input * self.config.input_normalization_scale

        if self.config.input_normalization == "asinh":
            dct_image = torch.sinh(scaled_output)
        elif self.config.input_normalization == "no_normalization":
            dct_image = scaled_output
        else:
            raise Exception("Invalid input norm")
        image = idct2(dct_image)
        return image

    def get_dataloader(self):
        trainset = self.get_trainset()

        padding_transform = Compose(
            [
                Lambda(lambda t: self.transform_to_scaled_fourier(t)),
                Pad(self.config.pad),
            ]
        )

        # Create an instance of the DCTScaledDataset class and pass in the Fashion MNIST dataset
        dct_data = self.get_dataset(trainset, transform=padding_transform)

        # trainloader = DataLoader(dct_data, batch_size=self.config.batch_size, shuffle=not self.config.overfit, pin_memory = True, num_workers = 0)
        trainloader = DataLoader(
            dct_data,
            batch_size=self.config.batch_size,
            shuffle=not self.config.overfit,
            pin_memory=True,
        )

        # if self.config.cache_dataset:
        #     logging.info("Caching dataset")
        #     for index, data in enumerate(tqdm(trainloader)):
        #         pass
        #         # print(len(trainloader.dataset.cached_data))

        #     logging.info("Dataset cached")
        #     trainloader.dataset.set_use_cache(use_cache=True)

        # trainloader.num_workers = 0
        # trainloader.prefetch_factor = 2

        return trainloader

    def get_model(self):
        if self.config.unet == "ddpm_unet":
            logging.info(f"Using ddpm_unetUnet")
            model = ddpm_unetUnet(
                resolution=self.config.image_size,
                T=self.config.num_train_timesteps,
                in_channels=self.config.channels,
                ch=128,
                ch_mult=[1, 2, 2, 2],
                # ch_mult=[1, 2, 2],
                attn=[1],
                num_res_blocks=2,
                dropout=0.1,
            )
        elif self.config.unet == "DiT":
            model = DiT_models[self.config.DiT_model](
                input_size=self.config.image_size,
                in_channels=self.config.channels,
                num_classes=self.config.image_size,
            )
        else:
            raise ValueError("Unet type not recognized")
        logging.info(f"number of parameters: {count_parameters(model)}")
        return model

    def get_optimizer(self):
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.9)
        # optimizer = Adam(self.model.parameters(), lr=1e-3)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def get_scheduler(self):
        if self.config.fermi_schedule == "linear":
            logging.info("Using linear fermi schedule")
            noise_scheduler = fourier_scheduler.FourierScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.image_size,
            )
        elif self.config.fermi_schedule == "exp":
            logging.info("Using exp fermi schedule")
            noise_scheduler = fourier_exp_scheduler.FourierExpScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                temperature=self.config.temperature,
                fermi_start=self.config.fermi_start,
                fermi_end=self.config.fermi_end,
                image_size=self.config.image_size,
                fermi_muliplier=self.config.fermi_muliplier,
            )

        return noise_scheduler

    def get_loss(self, timesteps, noise, noise_pred):
        loss_wrappers = self.scheduler.get_loss_wrapper(
            timesteps,
            noise.shape,
            self.config.t2,
            func=self.config.loss_wrapper_function,
            normalize=self.config.normalize_loss_wrapper,
        )
        loss = F.mse_loss(noise_pred * loss_wrappers, noise * loss_wrappers)
        return loss

    def get_pipeline(self, model, scheduler):
        pipeline = fourier_pipeline.FourierPipeline(
            unet=model,
            scheduler=scheduler,
            denoise_algo=self.config.denoise_algo,
            loss_type=self.config.loss_type,
            ddim_sigma=self.config.ddim_sigma,
        )
        return pipeline

    @torch.no_grad()
    def get_noise_images(self, clean_images, noise, timesteps):
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images

    def get_noise(self, shape, device):
        return torch.randn(shape).to(device)

    def get_num_train_timesteps(self):
        return self.scheduler.num_train_timesteps

    def get_num_eval_timesteps(self):
        pass

    @torch.no_grad()
    def save_image(self, images, path, nrow=8, tag=None, step=None, index=None):
        arr = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0).cpu()
        print(arr.shape)
        if index is not None:
            arr = torch.abs(arr[:, :, index])
        fig, ax = plt.subplots()
        im = ax.imshow(arr)  # Here make an AxesImage rather than contour
        fig.colorbar(im)
        vmax = torch.max(arr)
        vmin = torch.min(arr)
        im.set_clim(vmin, vmax)
        fig.savefig(path)
        # self.writer.add_image(tag, arr, step, dataformats='HWC')

    @torch.no_grad()
    def sample_and_save(self, epoch, _step, clean_images=True):
        # save generated images
        seed = 0
        nrow = int(math.sqrt(self.config.eval_batch_size))
        if _step != 0 and _step % self.config.save_and_sample_every == 0:
            if (
                clean_images is not None
                and self.config.save_input_samples
                and _step // self.config.save_and_sample_every == 1
            ):
                self.save_image(
                    clean_images[: self.config.eval_batch_size],
                    str(self.config.results_folder / f"input-{_step}.png"),
                    nrow=nrow,
                    tag="Input",
                    step=_step,
                    index=0,
                )

            images, fourier = self.sample(
                self.config.eval_batch_size, ema=False, seed=seed
            )
            images_ema, fourier_ema = self.sample(
                self.config.eval_batch_size, ema=True, seed=seed
            )
            logging.info(f"Step: {_step}, Generated images: {images.shape}")
            if self.config.save_fourier_samples:
                self.save_image(
                    fourier,
                    str(self.config.results_folder / f"fourier-sample-{_step}.png"),
                    nrow=nrow,
                    tag="Fourier",
                    step=_step,
                    index=0,
                )
                self.save_image(
                    fourier_ema,
                    str(self.config.results_folder / f"fourier-sample-{_step}-ema.png"),
                    nrow=nrow,
                    tag="Fourier",
                    step=_step,
                    index=0,
                )
            self.save_image(
                images,
                str(self.config.results_folder / f"sample-{_step}.png"),
                nrow=nrow,
                tag="Inverted",
                step=_step,
            )
            self.save_image(
                images_ema,
                str(self.config.results_folder / f"sample-{_step}-ema.png"),
                nrow=nrow,
                tag="Inverted",
                step=_step,
            )

        if _step != 0 and _step % self.config.save_state_every == 0:
            self.save_state(_step)

    def sample(self, num, ema=False, seed=0, return_all_steps=False):
        if ema:
            pipeline = self.pipeline_ema
        else:
            pipeline = self.pipeline
        print("Num inference steps: ", self.config.num_inference_steps)
        images, fourier = pipeline(
            batch_size=num,
            generator=torch.manual_seed(seed),
            num_inference_steps=self.config.num_inference_steps,
            pad=self.config.pad,
            return_image_and_fourier=True,
            transform_from_scaled_fourier=self.transform_from_scaled_fourier,
            return_all=return_all_steps,
            # convert_to_image_space=False
        )
        # print(f"Foureir max: {torch.max(fourier)}, Fourier min: {torch.min(fourier)}")
        # print(f"Image max: {torch.max(images)}, Image min: {torch.min(images)}")

        return images, fourier

    def save_state(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        data = {
            "step": step,
            "model": unwrapped_model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "ema": self.ema_model.state_dict(),
            # 'ema': self.ema.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }
        torch.save(data, f"{self.config.results_folder}/{f'model-{step}.pt'}")

    def load_state(self):
        if self.config.checkpoint_path is not None:
            device = self.config.device
            logging.info(f"Loading checkpoint from {self.config.checkpoint_path}")
            data = torch.load(self.config.checkpoint_path, map_location=device)
            if self.config.ddpm_unet_checkpoint:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.load_state_dict(data["net_model"])
                logging.info(f"Loaded model")
                if self.accelerator.is_main_process:
                    self.ema_model.load_state_dict(data["ema_model"])
                    logging.info(f"Loaded ema")
                self.optimizer.load_state_dict(data["optim"])
                logging.info(f"Loaded optimizer")
                self.init_step = data["step"]
                logging.info(f"Loaded step {self.init_step}")
            else:
                logging.info(f"Loading checkpoint from {self.config.checkpoint_path}")
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.load_state_dict(data["model"])
                logging.info(f"Loaded model")
                self.ema_model.load_state_dict(data["ema"])
                logging.info(f"Loaded ema")
                self.optimizer.load_state_dict(data["opt"])
                logging.info(f"Loaded optimizer")
                self.init_step = data["step"]
                logging.info(f"Loaded step {self.init_step}")

    def get_train_timesteps(self, start, end, batch_size, device):
        return torch.randint(start, end, (batch_size,), device=device).long()

    def modify_grads(self, model):
        self.accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

    def run(self):
        # torch.autograd.set_detect_anomaly(False)
        _step = self.init_step
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        model = self.model
        device = self.config.device
        # ema = self.ema
        ema_model = self.ema_model
        single_batch_iterations = 1000000 if self.config.overfit else 1
        for epoch in range(self.config.epochs):
            logging.info(f"epoch: {epoch}")
            for step, data in enumerate(self.dataloader, 0):
                for l in range(single_batch_iterations):
                    _step += 1
                    # get the inputs; data is a list of [inputs, labels]
                    clean_images, labels = data[0].to(device), data[1].to(device)

                    batch_size = data[0].shape[0]

                    # # Algorithm 1 line 3: sample t uniformally for every example in the batch (algo is in the blogpost)
                    timesteps = self.get_train_timesteps(
                        0, self.get_num_train_timesteps(), batch_size, device
                    )

                    noise = self.get_noise(clean_images.shape, clean_images.device)

                    # Add noise to the clean images according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_images = self.get_noise_images(clean_images, noise, timesteps)

                    noise_pred = model(noisy_images, timesteps)

                    loss = self.get_loss(timesteps, noise, noise_pred)

                    # loss.backward()

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.modify_grads(model)

                    optimizer.step()
                    lr_scheduler.step()
                    # ema.update()
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        simple_ema(unwrapped_model, ema_model, self.config.ema_decay)
                    if self.accelerator.is_main_process:
                        if _step % 100 == 0:
                            logging.info(f"Step: {_step}, Loss: {loss.item()}")
                            self.writer.add_scalar("Loss/train", loss, _step)
                            self.writer.add_scalar(
                                "Learning Rate", lr_scheduler.get_last_lr()[0], _step
                            )

                        self.sample_and_save(epoch, _step, clean_images)

                    optimizer.zero_grad()

                if _step == self.config.max_steps:
                    break
            if _step == self.config.max_steps:
                break

        self.save_state("final")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=str, default="", help="named identifier for the run"
    )
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    return parser


def parse_config(parser):
    args = parser.parse_args()
    config = {}
    if args.config != "":
        config_json = json.load(open(args.config))
        config_json["run_name"] = args.n
    config = Config(**config_json)

    return config


if __name__ == "__main__":
    parser = get_parser()
    config = parse_config(parser)
    runner = FourierRunner(config)
    runner.run()

# %%
