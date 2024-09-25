import torch_dct as dct
from scipy import fftpack
from einops import rearrange
import torch
import torch.nn.functional as F


from stats.mean_std_Cifar10 import (
    train_upscaled_static_mean as train_upscaled_static_mean_cifar10,
    train_upscaled_static_std as train_upscaled_static_std_cifar10,
)


from stats.mean_std_Cifar10_P32 import (
    train_upscaled_static_mean as train_upscaled_static_mean_cifar10_p32,
    train_upscaled_static_std as train_upscaled_static_std_cifar10_p32,
)

from stats.mean_std_Cifar10_P4 import (
    train_upscaled_static_mean as train_upscaled_static_mean_cifar10_p4,
    train_upscaled_static_std as train_upscaled_static_std_cifar10_p4,
)

from stats.mean_std_Cifar10_P2 import (
    train_upscaled_static_mean as train_upscaled_static_mean_cifar10_p2,
    train_upscaled_static_std as train_upscaled_static_std_cifar10_p2,
)

from stats.mean_std_Cifar10_P1 import (
    train_upscaled_static_mean as train_upscaled_static_mean_cifar10_p1,
    train_upscaled_static_std as train_upscaled_static_std_cifar10_p1,
)


from stats.mean_std_CelebA import (
    train_upscaled_static_mean as train_upscaled_static_mean_CelebA,
    train_upscaled_static_std as train_upscaled_static_std_CelebA,
)
from stats.mean_std_CelebA_P4 import (
    train_upscaled_static_mean as train_upscaled_static_mean_CelebA_p4,
    train_upscaled_static_std as train_upscaled_static_std_CelebA_p4,
)


from stats.mean_std_FMNIST import (
    train_upscaled_static_mean as train_upscaled_static_mean_FMNIST,
    train_upscaled_static_std as train_upscaled_static_std_FMNIST,
)

from stats.mean_std_ImageNette_DCTNet_P8 import (
    train_upscaled_static_mean as train_upscaled_static_mean_ImageNette8,
    train_upscaled_static_std as train_upscaled_static_std_ImageNette8,
)
from stats.mean_std_Cifar10_DCTNet_P8 import (
    train_upscaled_static_mean as train_upscaled_static_mean_CIFAR10_DCTNet_8,
    train_upscaled_static_std as train_upscaled_static_std_CIFAR10_DCTNet_8,
)


from stats.mean_std_CelebA_DCTNet_P8 import (
    train_upscaled_static_mean as train_upscaled_static_mean_CelebA_DCTNet_8,
    train_upscaled_static_std as train_upscaled_static_std_CelebA_DCTNet_8,
)

from stats.mean_std_LSUN_Bedroom_DCTNet_P8 import (
    train_upscaled_static_mean as train_upscaled_static_mean_LSUN_Bedroom_DCTNet_8,
    train_upscaled_static_std as train_upscaled_static_std_LSUN_Bedroom_DCTNet_8,
)


from stats.mean_std_FashionMNIST_DCTNet_P8 import (
    train_upscaled_static_mean as train_upscaled_static_mean_FashionMNIST_DCTNet_8,
    train_upscaled_static_std as train_upscaled_static_std_FashionMNIST_DCTNet_8,
)


def dct2(a):
    return dct.dct_2d(a, "ortho")


# implement 2D IDCT
def idct2(a):
    return dct.idct_2d(a, "ortho")


def scipy_dct2(x):
    return fftpack.dct(fftpack.dct(x, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def scipy_idct2(x):
    return fftpack.idct(fftpack.idct(x, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def get_dataset_stats_v7(dataset, patch_size, dummy=False):
    patch_size = str(patch_size)
    mean_stats = {
        "ImageNette": {
            "8": train_upscaled_static_mean_ImageNette8,
        },
        "CIFAR10": {
            "8": train_upscaled_static_mean_CIFAR10_DCTNet_8,
        },
        "CelebA": {
            "8": train_upscaled_static_mean_CelebA_DCTNet_8,
        },
        "LSUN_Bedroom": {"8": train_upscaled_static_mean_LSUN_Bedroom_DCTNet_8},
        "FashionMNIST": {"8": train_upscaled_static_mean_FashionMNIST_DCTNet_8},
    }

    std_stats = {
        "ImageNette": {
            "8": train_upscaled_static_std_ImageNette8,
        },
        "CIFAR10": {
            "8": train_upscaled_static_std_CIFAR10_DCTNet_8,
        },
        "CelebA": {
            "8": train_upscaled_static_std_CelebA_DCTNet_8,
        },
        "LSUN_Bedroom": {"8": train_upscaled_static_std_LSUN_Bedroom_DCTNet_8},
        "FashionMNIST": {"8": train_upscaled_static_std_FashionMNIST_DCTNet_8},
    }

    mean, std = mean_stats[dataset][patch_size], std_stats[dataset][patch_size]
    if dummy:
        return torch.zeros(len(mean)), torch.ones(len(std))
    else:
        return mean, std


def get_dataset_stats(dataset, patch_size, dummy=False):
    patch_size = str(patch_size)
    mean_stats = {
        "CIFAR10": {
            "32": train_upscaled_static_mean_cifar10_p32,
            "8": train_upscaled_static_mean_cifar10,
            "4": train_upscaled_static_mean_cifar10_p4,
            "2": train_upscaled_static_mean_cifar10_p2,
            "1": train_upscaled_static_mean_cifar10_p1,
        },
        "CelebA": {
            "8": train_upscaled_static_mean_CelebA,
            "4": train_upscaled_static_mean_CelebA_p4,
            "64": torch.zeros(12288),
        },
        "FashionMNIST": {
            "7": train_upscaled_static_mean_FMNIST,
            "28": torch.zeros(784),
        },
        "ImageNette": {"8": torch.zeros(192)},
        "CIFAR10_64": {"8": torch.zeros(192), "4": torch.zeros(48)},
    }

    std_stats = {
        "CIFAR10": {
            "32": train_upscaled_static_std_cifar10_p32,
            "8": train_upscaled_static_std_cifar10,
            "4": train_upscaled_static_std_cifar10_p4,
            "2": train_upscaled_static_std_cifar10_p2,
            "1": train_upscaled_static_std_cifar10_p1,
        },
        "CelebA": {
            "8": train_upscaled_static_std_CelebA,
            "4": train_upscaled_static_std_CelebA_p4,
            "64": torch.ones(12288),
        },
        "FashionMNIST": {
            "7": train_upscaled_static_std_FMNIST,
            "28": torch.ones(784),
        },
        "ImageNette": {"8": torch.ones(192)},
        "CIFAR10_64": {"8": torch.ones(192), "4": torch.ones(48)},
    }

    mean, std = mean_stats[dataset][patch_size], std_stats[dataset][patch_size]
    if dummy:
        return torch.zeros(len(mean)), torch.ones(len(std))
    else:
        return mean, std


def patchify(img, patch_size):
    return rearrange(
        img,
        "b c (h2 h1) (w2 w1) -> b (c h2 w2) h1 w1",
        h1=patch_size,
        w1=patch_size,
    )


def unpatchify(patched_image, num_patches_per_dim):
    return rearrange(
        patched_image,
        "b (c h2 w2) h1 w1 -> b c (h2 h1) (w2 w1)",
        h2=num_patches_per_dim,
        w2=num_patches_per_dim,
    )


def group_dct_components(patched_image, patch_size, num_patches_per_dim, num_channels):
    if patch_size == 32:
        return patched_image
    return rearrange(
        patched_image,
        "b (c1 c2 c3) (h2 h1) (w2 w1) -> b (c1 h1 w1) (c2 h2) (c3 w2)",
        h1=patch_size,
        w1=patch_size,
        c1=num_channels,
        c2=num_patches_per_dim,
        h2=1,
        w2=1,
    )


def ungroup_dct_components(grouped_dct_components, patch_size, num_patches_per_dim):
    if patch_size == 32:
        return grouped_dct_components
    return rearrange(
        grouped_dct_components,
        "b (c1 h1 w1) (c2 h2) (c3 w2) -> b (c1 c2 c3) (h2 h1) (w2 w1)",
        h1=patch_size,
        w1=patch_size,
        c2=num_patches_per_dim,
        c3=num_patches_per_dim,
        h2=1,
        w2=1,
    )


def ungroup_components_stat(
    components_stat, input_channels, patch_size, num_patches_per_dim
):
    reshaped_stat = torch.tensor(components_stat).view(
        input_channels, patch_size, patch_size
    )
    reshaped_stat = reshaped_stat.unsqueeze(1).repeat(1, num_patches_per_dim**2, 1, 1)
    reshaped_stat = reshaped_stat.view(
        num_patches_per_dim**2 * input_channels, patch_size, patch_size
    )
    # reshaped_stat = reshaped_stat.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return reshaped_stat

