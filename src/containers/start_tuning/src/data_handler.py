"""Data handling for SDXL LoRA training."""

import logging
import os
import random
from typing import Dict, List, Any

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer

from config import TrainingConfig
from utils_train import tokenize_prompt

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles dataset loading and preprocessing for SDXL LoRA training."""
    
    def __init__(
        self, 
        config: TrainingConfig,
        tokenizer_one: AutoTokenizer,
        tokenizer_two: AutoTokenizer
    ):
        self.config = config
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.dataset = None
        self.train_dataset = None
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms for training."""
        # Get interpolation method
        interpolation = getattr(
            transforms.InterpolationMode, 
            self.config.image_interpolation_mode.upper(), 
            None
        )
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode: {self.config.image_interpolation_mode}")
        
        # Setup transforms
        self.train_resize = transforms.Resize(self.config.resolution, interpolation=interpolation)
        self.train_crop = (
            transforms.CenterCrop(self.config.resolution) 
            if self.config.center_crop 
            else transforms.RandomCrop(self.config.resolution)
        )
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def load_dataset(self):
        """Load the training dataset."""
        logger.info("Loading dataset...")
        
        if self.config.dataset_name is not None:
            logger.info(f"Loading dataset from hub: {self.config.dataset_name}")
            self.dataset = load_dataset(
                self.config.dataset_name, 
                self.config.dataset_config_name, 
                cache_dir=self.config.cache_dir, 
                data_dir=self.config.train_data_dir
            )
        else:
            logger.info(f"Loading dataset from directory: {self.config.train_data_dir}")
            data_files = {}
            if self.config.train_data_dir is not None:
                data_files["train"] = os.path.join(self.config.train_data_dir, "**")
            self.dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=self.config.cache_dir,
            )
        
        self._validate_dataset_columns()
    
    def _validate_dataset_columns(self):
        """Validate that required columns exist in the dataset."""
        column_names = self.dataset["train"].column_names
        
        # Validate image column
        if self.config.image_column not in column_names:
            if len(column_names) > 0:
                logger.warning(f"Image column '{self.config.image_column}' not found. Using '{column_names[0]}'")
                self.config.image_column = column_names[0]
            else:
                raise ValueError("No columns found in dataset")
        
        # Validate caption column
        if self.config.caption_column not in column_names:
            if len(column_names) > 1:
                logger.warning(f"Caption column '{self.config.caption_column}' not found. Using '{column_names[1]}'")
                self.config.caption_column = column_names[1]
            else:
                raise ValueError(f"Caption column '{self.config.caption_column}' not found in dataset")
    
    def prepare_dataset(self, accelerator):
        """Prepare the dataset for training."""
        logger.info("Preparing dataset...")
        
        with accelerator.main_process_first():
            if self.config.max_train_samples is not None:
                logger.info(f"Limiting training samples to {self.config.max_train_samples}")
                self.dataset["train"] = (
                    self.dataset["train"]
                    .shuffle(seed=self.config.seed)
                    .select(range(self.config.max_train_samples))
                )
            
            # Set the training transforms
            self.train_dataset = self.dataset["train"].with_transform(
                self._preprocess_train, 
                output_all_columns=True
            )
        
        logger.info(f"Training dataset size: {len(self.train_dataset)}")
    
    def _tokenize_captions(self, examples: Dict, is_train: bool = True) -> tuple:
        """Tokenize captions using both tokenizers."""
        captions = []
        for caption in examples[self.config.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # Take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column '{self.config.caption_column}' should contain strings or lists of strings."
                )
        
        tokens_one = tokenize_prompt(self.tokenizer_one, captions)
        tokens_two = tokenize_prompt(self.tokenizer_two, captions)
        return tokens_one, tokens_two
    
    def _preprocess_train(self, examples: Dict) -> Dict:
        """Preprocess training examples."""
        images = [image.convert("RGB") for image in examples[self.config.image_column]]
        
        # Image augmentation
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        
        for image in images:
            original_sizes.append((image.height, image.width))
            image = self.train_resize(image)
            
            # Random horizontal flip
            if self.config.random_flip and random.random() < 0.5:
                image = self.train_flip(image)
            
            # Cropping
            if self.config.center_crop:
                y1 = max(0, int(round((image.height - self.config.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - self.config.resolution) / 2.0)))
                image = self.train_crop(image)
            else:
                y1, x1, h, w = self.train_crop.get_params(
                    image, (self.config.resolution, self.config.resolution)
                )
                image = crop(image, y1, x1, h, w)
            
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = self.train_transforms(image)
            all_images.append(image)
        
        # Update examples
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        
        # Tokenize captions
        tokens_one, tokens_two = self._tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        
        # Add filenames for debugging if available
        if self.config.debug_loss:
            fnames = [
                os.path.basename(image.filename) 
                for image in examples[self.config.image_column] 
                if hasattr(image, 'filename') and image.filename
            ]
            if fnames:
                examples["filenames"] = fnames
        
        return examples
    
    @staticmethod
    def collate_fn(examples: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for the dataloader."""
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        
        result = {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }
        
        # Add filenames if available for debugging
        filenames = [
            example["filenames"] 
            for example in examples 
            if "filenames" in example
        ]
        if filenames:
            result["filenames"] = filenames
        
        return result
    
    def create_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        logger.info("Creating dataloader...")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.dataloader_num_workers,
        )