import json
from matplotlib import pyplot as plt
import tqdm
import sys
import torch
import os, PIL.Image
import math
import argparse
import logging

sys.path.insert(0, os.getcwd())
from config import Config
from ddpm_process import DDPMRunner
from fourier_process import FourierRunner
from fourier_process import get_parser, parse_config
import numpy as np
import matplotlib.animation as animation
import torchvision
from fddm_process import FDDMRunner


def make_gif(images, fouriers):
    n_timestesp = len(images)
    fig, ax = plt.subplots(figsize=(10, 20), nrows=2, ncols=1, dpi=30)
    ims = []

    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    # gif_images = images[0][:16,0].unsqueeze(1)
    gif_images = [torch.from_numpy(image).permute(2, 0, 1) for image in images[0][:16]]
    arr = torchvision.utils.make_grid(gif_images, nrow=4).permute(1, 2, 0)
    cv0 = arr
    im = ax[0].imshow(cv0, animated=True)  # Here make an AxesImage rather than contour
    cb = fig.colorbar(im)
    tx = ax[0].set_title("Timestep 0")

    gif_fouriers = [
        torch.from_numpy(image).permute(2, 0, 1) for image in fouriers[0][:16]
    ]
    arr = torchvision.utils.make_grid(gif_fouriers, nrow=4).permute(1, 2, 0)
    cv0 = arr
    # im_f = ax[1].imshow(cv0, animated=True, vmin=-40, vmax=40) # Here make an AxesImage rather than contour
    im_f = ax[1].imshow(
        cv0, animated=True
    )  # Here make an AxesImage rather than contour
    cb_f = fig.colorbar(im_f)
    tx_f = ax[1].set_title("Timestep 0")

    def animate(i):
        # print("1-: ", i)
        gif_images = [
            torch.from_numpy(image).permute(2, 0, 1) for image in images[i][:16]
        ]
        arr = torchvision.utils.make_grid(gif_images, nrow=4).permute(1, 2, 0)
        # print("2-: ", i)
        # print(arr.shape)
        vmax = torch.max(arr)
        vmin = torch.min(arr)
        im.set_clim(vmin, vmax)
        im.set_data(arr)
        tx.set_text("Timestep {0}".format(i))

        gif_fouriers = [
            torch.from_numpy(image).permute(2, 0, 1) for image in fouriers[i][:16]
        ]
        arr = torchvision.utils.make_grid(gif_fouriers, nrow=4).permute(1, 2, 0)
        # print("2-: ", i)
        # print(arr.shape)
        vmax = torch.max(arr)
        vmin = torch.min(arr)
        im_f.set_clim(vmin, vmax)
        # arr = (arr - vmin) / (vmax - vmin)
        im_f.set_data(arr)
        tx_f.set_text("Timestep {0}".format(i))
        # print("3-: ", i)
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=tqdm.tqdm(range(n_timestesp)),
        interval=math.floor(20000 / n_timestesp),
    )

    return ani


def parse_config(args):
    config = {}
    remove_keys = ["tensorboard_folder", "unique_id", "unique_name"]
    if args.config != "":
        config_json = json.load(open(args.config))
        for key in remove_keys:
            if key in config_json:
                del config_json[key]
        config_json["run_name"] = args.n
    config = Config(**config_json)

    return config


class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_runner(
    config_base_dir,
    output_name,
    checkpoint_name,
    ema,
    config_full_path=None,
    config_override=None,
):

    output_path = f"generated_samples_for_score/{output_name}"
    print(f"output_path: {output_path}")

    args = {
        "config": config_full_path or f"{config_base_dir}/config.json",
        "ema": ema,
        "n": output_path,
    }
    args = Dict2Class(args)
    config = parse_config(args)
    runner_name = config.process_name

    if runner_name == "ddpm":
        runner_class = DDPMRunner
    elif runner_name == "fourier":
        runner_class = FourierRunner
    elif runner_name == "fddm":
        runner_class = FDDMRunner
    else:
        raise "Runner not supported, please use regular or fourier"


    if checkpoint_name:
        config.checkpoint_path = f"{config_base_dir}/{checkpoint_name}"

    if not config.checkpoint_path:
        raise "Please provide a checkpoint path"

    if config_override:
        for key, value in config_override.items():
            print(f"Overriding {key} with {value}")
            setattr(config, key, value)

    print(f"config.checkpoint_path: {config.checkpoint_path}")

    runner = runner_class(config)
    logging.info(runner_class)
    os.makedirs(f"{config.results_folder}/img", exist_ok=True)
    if config.save_fourier_samples:
        os.makedirs(f"{config.results_folder}/fourier", exist_ok=True)
    return runner, config


def gen_samples(runner, num_samples, ema, config):
    logging.info(f"Ema: {ema}")
    nb = math.ceil(num_samples / config.eval_batch_size)
    for ib in tqdm.tqdm(range(nb), desc="Batch"):
        all_steps_img, all_step_fouriers = runner.sample(
            config.eval_batch_size, ema=ema, seed=ib, return_all_steps=True
        )
        print("----------------------")
        print(
            "Image: ",
            torch.max(torch.stack(all_steps_img)),
            torch.min(torch.stack(all_steps_img)),
        )
        print("----------------------")
        print(
            "Fourier: ",
            torch.max(torch.stack(all_step_fouriers)[-1]),
            torch.min(torch.stack(all_step_fouriers)[-1]),
        )
        all_steps_img = [torch.clamp(images, 0, 1) for images in all_steps_img]
        all_steps_img = [images.detach().cpu() for images in all_steps_img]
        all_steps_img = [images.permute(0, 2, 3, 1) for images in all_steps_img]
        if config.save_fourier_samples:
            # all_step_fouriers = [dct2(fouriers) for fouriers in all_step_fouriers]
            all_step_fouriers = [fouriers for fouriers in all_step_fouriers]
            all_step_fouriers = [
                fouriers.detach().cpu() for fouriers in all_step_fouriers
            ]
            all_step_fouriers = [
                fouriers.permute(0, 2, 3, 1) for fouriers in all_step_fouriers
            ]
        all_steps_img = [images * 255 for images in all_steps_img]
        all_steps_img = [images.numpy().astype(np.uint8) for images in all_steps_img]
        last_step_img = all_steps_img[-1]
        if config.save_fourier_samples:
            all_step_fouriers = [fouriers.numpy() for fouriers in all_step_fouriers]
            last_step_fouriers = all_step_fouriers[-1]
        print(f"last_step_img, {last_step_img.shape}")
        for i in range(last_step_img.shape[0]):
            img_to_save = None
            mode = None
            if last_step_img[i].shape[2] == 1:
                mode = "L"
                img_to_save = last_step_img[i][:, :, 0]
            elif last_step_img[i].shape[2] == 3:
                mode = "RGB"
                img_to_save = last_step_img[i]
            else:
                raise "Not supported"
            PIL.Image.fromarray(img_to_save, mode=mode).save(
                f"{config.results_folder}/img/{ib*config.eval_batch_size+i}.png"
            )
            if config.save_fourier_samples:
                fourier_to_save = last_step_fouriers[i]
                fig, ax = plt.subplots()
                # fig.set_size_inches(32, 32)
                ax.set_axis_off()
                im = ax.imshow(
                    fourier_to_save, aspect="equal"
                )  # Here make an AxesImage rather than contour
                fig.colorbar(im)
                vmax = np.max(fourier_to_save)
                vmin = np.min(fourier_to_save)
                im.set_clim(vmin, vmax)
                fig.savefig(
                    f"{config.results_folder}/fourier/{ib*config.eval_batch_size+i}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()


def main(args):
    config_override = {
        "save_fourier_samples": False,
        "num_inference_steps": 1000,
        "eval_batch_size": 200,
    }
    runner, config = get_runner(
        config_base_dir=args.config_base_dir,
        output_name=args.n,
        checkpoint_name=args.checkpoint_name,
        ema=args.ema,
        config_override=config_override,
    )
    gen_samples(runner=runner, num_samples=args.num_samples, config=config, ema=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = get_parser()
    parser.add_argument("--num_samples", type=int, help="number of samples to generate")
    parser.add_argument("-n", required=True, type=str, help="output name")
    # parser.add_argument("--checkpoint_name", default="model-final.pt" ,type=str, help='checkpoint name')
    parser.add_argument(
        "--checkpoint_name", default="model-790000.pt", type=str, help="checkpoint name"
    )
    parser.add_argument(
        "--config_base_dir", required=True, type=str, help="dir to config folder"
    )
    parser.add_argument("--ema", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--gen_denoising_gif", default=False, action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    main(args)
