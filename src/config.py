from dataclasses import dataclass


@dataclass
class Config:
    device: int  # The device number for CUDA (e.g., 0, 1, 2, etc.)
    loss_wrapper_function: (
        str  # Type of loss wrapper function (options: 'sech' or 'noise_coeff')
    )
    normalize_loss_wrapper: bool  # Whether to normalize the loss wrapper
    channels: int  # Number of input channels for the model
    batch_size: int  # Batch size for training
    eval_batch_size: int  # Batch size for evaluation
    save_and_sample_every: int  # Number of steps between saving and sampling
    save_state_every: int  # Number of steps between saving model state
    epochs: int  # Number of training epochs
    num_train_timesteps: int  # Number of timesteps for training (T in the paper)
    max_steps: int  # Maximum number of training steps
    results_folder: str  # Path to save results
    fermi_start: float  # Starting value for Fermi schedule
    fermi_end: float  # Ending value for Fermi schedule
    temperature: float  # Temperature for noise scheduling (T_prime in the paper)
    pad: list  # Padding values for input images
    image_size: int  # Size of input images
    cache_dataset: bool  # Whether to cache the dataset
    run_name: str  # Identifier for the current run
    learning_rate: float  # Learning rate for optimizer
    lr_warmup_steps: int  # Number of warmup steps for learning rate scheduler
    dataset: str  # Name of the dataset (options: 'CelebA', 'ImageNette', 'CIFAR10_64', 'CIFAR10', 'FashionMNIST')
    ema_decay: float  # Decay rate for Exponential Moving Average (EMA)
    ema_update_every: int  # Number of steps between EMA updates
    save_input_samples: bool  # Whether to save input samples
    save_fourier_samples: bool  # Whether to save Fourier samples
    process_name: str  # Name of the processes (options: 'fddm', 'ddpm', 'fourier')
    t2: float = None  # t2 for sech wrapper function (not used in the paper)
    unet: str = "normal"  # Type of UNet model (options: 'normal', 'ddpm_unet', 'DiT')
    overfit: bool = False  # Whether to overfit on a single batch
    fermi_schedule: str = "linear"  # Type of Fermi schedule (options: 'linear', 'exp')
    fermi_muliplier: float = 1.0  # Multiplier for exponential Fermi schedule
    checkpoint_path: str = None  # Path to load checkpoint from
    input_normalization_scale: float = 1.0  # Scale factor for input normalization
    input_normalization: str = (
        None  # Type of input normalization (options: 'asinh', 'no_normalization')
    )
    lr_scheduler: str = (
        "cosine"  # Type of learning rate scheduler (options: 'identity', 'cosine')
    )
    denoise_algo: str = "1"  # Denoising algorithm
    loss_type: str = "original_img"  # Type of loss function
    DiT_model: str = None  # Specific DiT model to use (if unet is 'DiT')
    ddim_sigma: float = 0  # Sigma value for DDIM sampling
    runner: str = "fddm"  # Name of the runner script
    out_channels: int = 3  # Number of output channels
    patch_size: int = 8  # Patch size
    norm_freq_comps: bool = True  # Whether to normalize frequency components
    ddpm_unet_checkpoint: bool = (
        False  # Whether the checkpoint is from ddpm_unet repository
    )
    clip_sample: bool = False  # Whether to clip samples
    num_inference_steps: int = (
        None  # Number of inference steps (if None, uses num_train_timesteps)
    )
