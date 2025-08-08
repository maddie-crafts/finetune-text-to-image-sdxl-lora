"""Refactored SDXL LoRA training script with modular components."""

import logging
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import datasets
import transformers
import diffusers

from config import parse_args, TrainingConfig
from model_loader import ModelLoader
from data_handler import DataHandler
from trainer import SDXLLoRATrainer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_logging(accelerator: Accelerator):
    """Setup logging for different processes."""
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def validate_environment(config: TrainingConfig):
    """Validate the training environment."""
    # Check for MPS + bf16 incompatibility
    if torch.backends.mps.is_available() and config.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. "
            "Please use fp16 (recommended) or fp32 instead."
        )


def setup_accelerator(config: TrainingConfig) -> Accelerator:
    """Setup and return the accelerator."""
    logging_dir = Path(config.output_dir) / config.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, 
        logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )


def main(config: TrainingConfig):
    """Main training function."""
    logger.info("Starting SDXL LoRA training...")
    logger.info(f"Configuration: {config}")
    
    # Validate environment
    validate_environment(config)
    
    # Setup accelerator
    accelerator = setup_accelerator(config)
    
    # Setup logging
    setup_logging(accelerator)
    
    # Set seed for reproducibility
    if config.seed is not None:
        logger.info(f"Setting seed to {config.seed}")
        set_seed(config.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output directory: {config.output_dir}")
    
    try:
        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = ModelLoader(config, accelerator)
        
        # Load tokenizers and models
        tokenizer_one, tokenizer_two = model_loader.load_tokenizers()
        model_loader.load_models()
        
        # Initialize data handler
        logger.info("Initializing data handler...")
        data_handler = DataHandler(config, tokenizer_one, tokenizer_two)
        
        # Load and prepare dataset
        data_handler.load_dataset()
        data_handler.prepare_dataset(accelerator)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = SDXLLoRATrainer(config, accelerator, model_loader, data_handler)
        
        # Setup optimizer and scheduler
        trainer.setup_optimizer()
        
        # Prepare for training
        trainer.prepare_for_training()
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cli_main():
    """Command line interface main function."""
    try:
        # Parse arguments and create config
        config = parse_args()
        
        # Run training
        main(config)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    cli_main()