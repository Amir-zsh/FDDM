import json
import torch
from config import Config
from utils import dct2, idct2
from fddm_process import patchify, group_dct_components, FDDMRunner
from tqdm import tqdm

def parse_config(args):
    config = {}
    if args.config != "":
        config_json = json.load(open(args.config))
        config_json["run_name"] = args.n
    config = Config(
        **config_json
    )

    return config 

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def get_runner(runner_class,        
            config_path,
            output_name,
        ):

    output_path = f"dataset_stats/{output_name}"
    print(f"output_path: {output_path}")
    args = {
        "config": config_path,
        "n": output_path
    }

    args = Dict2Class(args)
    config = parse_config(args)
    runner = runner_class(config)
    return runner




def main(dataset_name, config_path):
    test_runner = get_runner(
                runner_class=FDDMRunner,
                config_path= config_path,
                output_name= dataset_name,
            )
    assert test_runner.config.dataset == dataset_name

    def get_mean_std_v1(loader):
        # VAR[X] = E[X**2] - E[X]**2
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        channels_min = 0
        channels_max = 0
        # Add tqdm for progress bar
        for data, _ in tqdm(loader):
            data = patchify(data, test_runner.config.patch_size)
            data = dct2(data)
            data = group_dct_components(data,
                    patch_size=test_runner.config.patch_size,
                    num_patches_per_dim=test_runner.config.image_size
                    // test_runner.config.patch_size,
                    num_channels=test_runner.config.channels
            )
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_min = min(channels_min, torch.min(data))
            channels_max = max(channels_max, torch.max(data))
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean**2)**0.5
        print(f"min: {channels_min}, max: {channels_max}")
        return mean, std


    patch_size = test_runner.config.patch_size

    mean, std = get_mean_std_v1(test_runner.dataloader)
    print(len(mean))
    with open(f'{test_runner.config.results_folder}/{test_runner.config.dataset}-{patch_size}.py', 'w') as f:
        f.write(f"train_upscaled_static_mean = {str(mean.tolist())}\n")
        f.write(f"train_upscaled_static_std = {str(std.tolist())}\n")


if __name__ == "__main__":
    dataset_name = "CelebA"
    # config_path = ""
    main(dataset_name, config_path)