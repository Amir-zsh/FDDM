import json
import argparse
from config import Config
from ddpm_process import DDPMRunner
from fddm_process import FDDMRunner
from fourier_process import FourierRunner


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
    runner_name = config.process_name
    if runner_name == "ddpm":
        runner_class = DDPMRunner
    elif runner_name == "fddm":
        runner_class = FDDMRunner
    elif runner_name == "fourier":
        runner_class = FourierRunner
    else:
        raise "Runner not supported, please use ddpm, fddm or fourier"
    runner = runner_class(config)
    runner.run()