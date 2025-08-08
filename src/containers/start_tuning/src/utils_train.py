"""Utility functions for SDXL LoRA training."""

import os
from typing import List, Tuple, Any, Optional

import torch
import numpy as np
from contextlib import nullcontext
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from transformers import PretrainedConfig, AutoTokenizer

from config import TrainingConfig

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def log_validation(
    pipeline: Any,
    args: TrainingConfig,
    accelerator: Accelerator,
    epoch: int,
    is_final_validation: bool = False,
) -> List[Any]:
    """Run validation and log generated images.
    
    Args:
        pipeline: The diffusion pipeline for generation
        args: Training configuration
        accelerator: Accelerator instance
        epoch: Current epoch number
        is_final_validation: Whether this is final validation
        
    Returns:
        List of generated images
    """
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    pipeline_args = {"prompt": args.validation_prompt}
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, 
    revision: Optional[str], 
    subfolder: str = "text_encoder"
) -> type:
    """Import text encoder class from model name or path.
    
    Args:
        pretrained_model_name_or_path: Path or name of the pretrained model
        revision: Model revision to use
        subfolder: Subfolder containing the text encoder
        
    Returns:
        Text encoder class
        
    Raises:
        ValueError: If model class is not supported
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def tokenize_prompt(tokenizer: AutoTokenizer, prompt: List[str]) -> torch.Tensor:
    """Tokenize prompts using the provided tokenizer.
    
    Args:
        tokenizer: The tokenizer to use
        prompt: List of prompts to tokenize
        
    Returns:
        Tokenized input IDs
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(
    text_encoders: List[Any], 
    tokenizers: Optional[List[AutoTokenizer]], 
    prompt: Optional[List[str]], 
    text_input_ids_list: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts using dual text encoders.
    
    Args:
        text_encoders: List of text encoders to use
        tokenizers: Optional list of tokenizers (if None, use text_input_ids_list)
        prompt: Optional list of prompts to encode
        text_input_ids_list: Optional pre-tokenized input IDs
        
    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
    """
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds