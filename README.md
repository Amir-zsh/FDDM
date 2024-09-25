# Frequency Domain Diffusion Model with Scale-Dependent Noise Schedule

This repository contains the implementation of our paper "Frequency Domain Diffusion Model with Scale-Dependent Noise Schedule". Our work introduces a novel diffusion process operating in the frequency domain, which leverages the sparse structure of frequency domain image representations and allows us to modify the training protocol, resulting in significant computation enhancements, without a significant drop in generated image quality.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Amir-zsh/FDDM.git
cd FDDM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download CelebA Dataset

```bash
cd data && sh ./download_celebA.sh && cd ..
```

## Usage

### Training

Navigate to the `src` directory:

```bash
cd src
```

Train the models using the following commands:

1. Train DDPM on CelebA:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch run.py --config ../configs/ddpm-CelebA.json --n ddpm_CelebA
```

2. Train FDDM with patch size 8 on CelebA:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch run.py --config ../configs/fddm-p8-CelebA.json --n fddm_p8_CelebA
```

### Generating Samples

After training, generate samples using:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch tools/generate_samples.py --config_base_dir ../results/fddm_p8_CelebA -n fddm_p8_CelebA_samples --num_samples 200
```

### Computing FID Score

Use `pytorch_fid` to compute the Fréchet Inception Distance (FID) score:

```bash
python -m pytorch_fid ../data/img_align_celeba_64x64 ../results/generated_samples_for_score/fddm_p8_CelebA_samples/img --device cuda:0
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@INPROCEEDINGS{10619452,
  author={Ziashahabi, Amir and Buyukates, Baturalp and Sheshmani, Artan and You, Yi-Zhuang and Avestimehr, Salman},
  booktitle={2024 IEEE International Symposium on Information Theory (ISIT)}, 
  title={Frequency Domain Diffusion Model with Scale-Dependent Noise Schedule}, 
  year={2024},
  pages={19-24},
  doi={10.1109/ISIT57864.2024.10619452}
}
```

## Contact

For any questions or issues, please open an issue on this repository or contact `ziashaha@usc.edu`.
