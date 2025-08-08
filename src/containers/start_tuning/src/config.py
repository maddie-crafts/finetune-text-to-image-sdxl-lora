"""Configuration classes for SDXL LoRA training."""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from torchvision import transforms


@dataclass
class TrainingConfig:
    """Configuration class for SDXL LoRA training parameters."""
    
    # Model parameters
    pretrained_model_name_or_path: str
    pretrained_vae_model_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    
    # Dataset parameters
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "text"
    max_train_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    
    # Training parameters
    output_dir: str = "sd-model-finetuned-lora"
    logging_dir: str = "logs"
    seed: Optional[int] = None
    resolution: int = 1024
    center_crop: bool = False
    random_flip: bool = False
    train_text_encoder: bool = False
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    
    # Validation parameters
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    
    # Checkpoint parameters
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    
    # Optimizer parameters
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    
    # Advanced parameters
    snr_gamma: Optional[float] = None
    prediction_type: Optional[str] = None
    noise_offset: float = 0.0
    allow_tf32: bool = False
    mixed_precision: Optional[str] = None
    dataloader_num_workers: int = 0
    local_rank: int = -1
    
    # Memory optimization
    enable_xformers_memory_efficient_attention: bool = False
    enable_npu_flash_attention: bool = False
    
    # LoRA parameters
    rank: int = 4
    
    # Logging parameters
    report_to: str = "tensorboard"
    debug_loss: bool = False
    
    # Image processing parameters
    image_interpolation_mode: str = "lanczos"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._set_derived_values()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.dataset_name is None and self.train_data_dir is None:
            raise ValueError("Need either a dataset name or a training folder.")
        
        if self.mixed_precision not in [None, "no", "fp16", "bf16"]:
            raise ValueError("Mixed precision must be one of: None, 'no', 'fp16', 'bf16'")
        
        if self.lr_scheduler not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}")
        
        # Validate image interpolation mode
        valid_modes = [f.lower() for f in dir(transforms.InterpolationMode) 
                      if not f.startswith("__") and not f.endswith("__")]
        if self.image_interpolation_mode.lower() not in valid_modes:
            raise ValueError(f"Invalid image_interpolation_mode: {self.image_interpolation_mode}")
    
    def _set_derived_values(self):
        """Set derived configuration values."""
        # Handle local rank from environment
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create TrainingConfig from argparse Namespace."""
        return cls(**vars(args))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for training configuration."""
    parser = argparse.ArgumentParser(description="SDXL LoRA fine-tuning script.")
    
    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., 'fp16').",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset from HuggingFace hub to train on.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes, truncate the number of training examples.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where downloaded models and datasets will be stored.",
    )
    
    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    
    # Validation parameters
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt used during validation to verify model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to generate during validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    
    # Checkpoint parameters
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    
    # Advanced parameters
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma for loss rebalancing.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type for training ('epsilon' or 'v_prediction').",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank.",
    )
    
    # Memory optimization
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether to use xformers.",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Whether to use npu flash attention.",
    )
    
    # LoRA parameters
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    
    # Logging parameters
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Integration to report results and logs to.",
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="Debug loss for each image if filenames are available.",
    )
    
    # Image processing parameters
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[f.lower() for f in dir(transforms.InterpolationMode) 
                if not f.startswith("__") and not f.endswith("__")],
        help="The image interpolation method for resizing.",
    )
    
    return parser


def parse_args(input_args=None) -> TrainingConfig:
    """Parse command line arguments and return TrainingConfig."""
    parser = create_argument_parser()
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return TrainingConfig.from_args(args)