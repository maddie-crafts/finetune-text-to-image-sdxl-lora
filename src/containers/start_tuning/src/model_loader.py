"""Model loading and configuration for SDXL LoRA training."""

import logging
from typing import Tuple, Optional

import torch
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import cast_training_params
from packaging import version
from peft import LoraConfig
from transformers import AutoTokenizer

from config import TrainingConfig
from utils_train import import_model_class_from_model_name_or_path

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and configuration of all models for SDXL LoRA training."""
    
    def __init__(self, config: TrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.weight_dtype = self._determine_weight_dtype()
        
        # Initialize models as None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
    
    def _determine_weight_dtype(self) -> torch.dtype:
        """Determine the weight dtype based on mixed precision setting."""
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype
    
    def load_tokenizers(self) -> Tuple[AutoTokenizer, AutoTokenizer]:
        """Load and return the tokenizers."""
        logger.info("Loading tokenizers...")
        
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.revision,
            use_fast=False,
        )
        
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.config.revision,
            use_fast=False,
        )
        
        return self.tokenizer_one, self.tokenizer_two
    
    def load_models(self):
        """Load all models required for training."""
        logger.info("Loading models...")
        
        # Load text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.config.pretrained_model_name_or_path, self.config.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.config.pretrained_model_name_or_path, 
            self.config.revision, 
            subfolder="text_encoder_2"
        )
        
        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.config.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=self.config.revision, 
            variant=self.config.variant
        )
        
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.config.pretrained_model_name_or_path, 
            subfolder="text_encoder_2", 
            revision=self.config.revision, 
            variant=self.config.variant
        )
        
        # Load VAE
        vae_path = (
            self.config.pretrained_model_name_or_path
            if self.config.pretrained_vae_model_name_or_path is None
            else self.config.pretrained_vae_model_name_or_path
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.config.pretrained_vae_model_name_or_path is None else None,
            revision=self.config.revision,
            variant=self.config.variant,
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path, 
            subfolder="unet", 
            revision=self.config.revision, 
            variant=self.config.variant
        )
        
        # Freeze non-trainable parameters
        self._freeze_models()
        
        # Move models to device and set dtype
        self._configure_model_devices()
        
        # Configure memory optimization
        self._configure_memory_optimization()
        
        # Add LoRA adapters
        self._add_lora_adapters()
        
        # Configure gradient checkpointing
        self._configure_gradient_checkpointing()
        
        # Cast training parameters to float32 if needed
        self._cast_training_parameters()
    
    def _freeze_models(self):
        """Freeze non-trainable model parameters."""
        logger.info("Freezing non-trainable parameters...")
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)
    
    def _configure_model_devices(self):
        """Move models to appropriate devices with correct dtypes."""
        logger.info("Moving models to device and setting dtypes...")
        
        # Move UNet to device
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        
        # VAE handling - keep in float32 unless using custom VAE
        if self.config.pretrained_vae_model_name_or_path is None:
            self.vae.to(self.accelerator.device, dtype=torch.float32)
        else:
            self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        # Text encoders
        self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
    
    def _configure_memory_optimization(self):
        """Configure memory optimization settings."""
        if self.config.enable_npu_flash_attention:
            if is_torch_npu_available():
                logger.info("Enabling NPU flash attention...")
                self.unet.enable_npu_flash_attention()
            else:
                raise ValueError("NPU flash attention requires torch_npu extensions and NPU devices.")
        
        if self.config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. "
                        "Please update xFormers to at least 0.0.17."
                    )
                logger.info("Enabling xFormers memory efficient attention...")
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly.")
        
        # Enable TF32 for faster training on Ampere GPUs
        if self.config.allow_tf32:
            logger.info("Enabling TF32...")
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to models."""
        logger.info("Adding LoRA adapters...")
        
        # UNet LoRA configuration
        unet_lora_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config)
        
        # Text encoder LoRA configuration (if enabled)
        if self.config.train_text_encoder:
            logger.info("Adding LoRA adapters to text encoders...")
            text_lora_config = LoraConfig(
                r=self.config.rank,
                lora_alpha=self.config.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.text_encoder_one.add_adapter(text_lora_config)
            self.text_encoder_two.add_adapter(text_lora_config)
    
    def _configure_gradient_checkpointing(self):
        """Configure gradient checkpointing if enabled."""
        if self.config.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing...")
            self.unet.enable_gradient_checkpointing()
            if self.config.train_text_encoder:
                self.text_encoder_one.gradient_checkpointing_enable()
                self.text_encoder_two.gradient_checkpointing_enable()
    
    def _cast_training_parameters(self):
        """Cast training parameters to appropriate dtype."""
        if self.accelerator.mixed_precision == "fp16":
            logger.info("Casting training parameters to float32...")
            models = [self.unet]
            if self.config.train_text_encoder:
                models.extend([self.text_encoder_one, self.text_encoder_two])
            cast_training_params(models, dtype=torch.float32)
    
    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameters."""
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        if self.config.train_text_encoder:
            params_to_optimize.extend(
                list(filter(lambda p: p.requires_grad, self.text_encoder_one.parameters()))
            )
            params_to_optimize.extend(
                list(filter(lambda p: p.requires_grad, self.text_encoder_two.parameters()))
            )
        return params_to_optimize
    
    @staticmethod
    def unwrap_model(model, accelerator: Accelerator):
        """Unwrap model from accelerator and compilation."""
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model