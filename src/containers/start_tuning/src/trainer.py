"""Training logic for SDXL LoRA training."""

import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm

from config import TrainingConfig
from data_handler import DataHandler
from model_loader import ModelLoader
from utils_train import encode_prompt, log_validation

logger = logging.getLogger(__name__)


class SDXLLoRATrainer:
    """Main trainer class for SDXL LoRA fine-tuning."""
    
    def __init__(
        self,
        config: TrainingConfig,
        accelerator: Accelerator,
        model_loader: ModelLoader,
        data_handler: DataHandler
    ):
        self.config = config
        self.accelerator = accelerator
        self.model_loader = model_loader
        self.data_handler = data_handler
        
        # Training state
        self.global_step = 0
        self.first_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        
        # Setup hooks
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup save and load hooks for checkpointing."""
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)
    
    def _save_model_hook(self, models, weights, output_dir):
        """Custom save hook for LoRA weights."""
        if not self.accelerator.is_main_process:
            return
        
        logger.info(f"Saving LoRA weights to {output_dir}")
        
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None
        
        for model in models:
            unwrapped_model = ModelLoader.unwrap_model(model, self.accelerator)
            
            if isinstance(unwrapped_model, type(ModelLoader.unwrap_model(self.model_loader.unet, self.accelerator))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(unwrapped_model, type(ModelLoader.unwrap_model(self.model_loader.text_encoder_one, self.accelerator))):
                text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(unwrapped_model, type(ModelLoader.unwrap_model(self.model_loader.text_encoder_two, self.accelerator))):
                text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            else:
                raise ValueError(f"Unexpected model type: {model.__class__}")
            
            # Remove weight to prevent double saving
            if weights:
                weights.pop()
        
        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )
    
    def _load_model_hook(self, models, input_dir):
        """Custom load hook for LoRA weights."""
        logger.info(f"Loading LoRA weights from {input_dir}")
        
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None
        
        while len(models) > 0:
            model = models.pop()
            
            if isinstance(model, type(ModelLoader.unwrap_model(self.model_loader.unet, self.accelerator))):
                unet_ = model
            elif isinstance(model, type(ModelLoader.unwrap_model(self.model_loader.text_encoder_one, self.accelerator))):
                text_encoder_one_ = model
            elif isinstance(model, type(ModelLoader.unwrap_model(self.model_loader.text_encoder_two, self.accelerator))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"Unexpected model type: {model.__class__}")
        
        # Load LoRA state dict
        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        
        # Load UNet LoRA weights
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading UNet LoRA: {unexpected_keys}")
        
        # Load text encoder LoRA weights if training text encoder
        if self.config.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_)
        
        # Cast training parameters to float32 if needed
        if self.accelerator.mixed_precision == "fp16":
            models = [unet_]
            if self.config.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            from diffusers.training_utils import cast_training_params
            cast_training_params(models, dtype=torch.float32)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer...")
        
        # Scale learning rate if requested
        if self.config.scale_lr:
            self.config.learning_rate = (
                self.config.learning_rate * 
                self.config.gradient_accumulation_steps * 
                self.config.train_batch_size * 
                self.accelerator.num_processes
            )
            logger.info(f"Scaled learning rate to {self.config.learning_rate}")
        
        # Get trainable parameters
        params_to_optimize = self.model_loader.get_trainable_parameters()
        
        # Choose optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                raise ImportError("To use 8-bit Adam, install bitsandbytes: pip install bitsandbytes")
        else:
            optimizer_class = torch.optim.AdamW
            logger.info("Using AdamW optimizer")
        
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        
        # Setup learning rate scheduler
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
        )
    
    def prepare_for_training(self):
        """Prepare models and data for training with accelerator."""
        logger.info("Preparing for training...")
        
        self.train_dataloader = self.data_handler.create_dataloader()
        
        if self.config.train_text_encoder:
            (
                self.model_loader.unet, 
                self.model_loader.text_encoder_one, 
                self.model_loader.text_encoder_two, 
                self.optimizer, 
                self.train_dataloader, 
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.model_loader.unet, 
                self.model_loader.text_encoder_one, 
                self.model_loader.text_encoder_two, 
                self.optimizer, 
                self.train_dataloader, 
                self.lr_scheduler
            )
        else:
            (
                self.model_loader.unet, 
                self.optimizer, 
                self.train_dataloader, 
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.model_loader.unet, 
                self.optimizer, 
                self.train_dataloader, 
                self.lr_scheduler
            )
        
        # Recalculate training steps
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        self.config.num_train_epochs = math.ceil(self.config.max_train_steps / num_update_steps_per_epoch)
    
    def _handle_checkpointing(self):
        """Handle checkpoint saving and cleanup."""
        if self.global_step % self.config.checkpointing_steps != 0:
            return
        
        if not (self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process):
            return
        
        # Handle checkpoint limit
        if self.config.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            if len(checkpoints) >= self.config.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                
                logger.info(f"Removing {len(removing_checkpoints)} old checkpoints")
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint_path = os.path.join(self.config.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint_path)
        
        # Save checkpoint
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def _compute_loss(self, model_pred, target, timesteps):
        """Compute the training loss."""
        if self.config.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute SNR-weighted loss
            snr = compute_snr(self.model_loader.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack(
                [snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            
            if self.model_loader.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.model_loader.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        return loss
    
    def _training_step(self, batch):
        """Execute a single training step."""
        with self.accelerator.accumulate(self.model_loader.unet):
            # Convert images to latent space
            pixel_values = batch["pixel_values"]
            if self.config.pretrained_vae_model_name_or_path is not None:
                pixel_values = pixel_values.to(dtype=self.model_loader.weight_dtype)
            
            model_input = self.model_loader.vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * self.model_loader.vae.config.scaling_factor
            
            if self.config.pretrained_vae_model_name_or_path is None:
                model_input = model_input.to(self.model_loader.weight_dtype)
            
            # Sample noise
            noise = torch.randn_like(model_input)
            if self.config.noise_offset:
                noise += self.config.noise_offset * torch.randn(
                    (model_input.shape[0], model_input.shape[1], 1, 1), 
                    device=model_input.device
                )
            
            # Sample timesteps
            bsz = model_input.shape[0]
            timesteps = torch.randint(
                0, self.model_loader.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=model_input.device
            ).long()
            
            # Add noise to model input
            noisy_model_input = self.model_loader.noise_scheduler.add_noise(model_input, noise, timesteps)
            
            # Compute time IDs
            def compute_time_ids(original_size, crops_coords_top_left):
                target_size = (self.config.resolution, self.config.resolution)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.model_loader.weight_dtype)
                return add_time_ids
            
            add_time_ids = torch.cat([
                compute_time_ids(s, c) 
                for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
            ])
            
            # Encode prompts
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=[self.model_loader.text_encoder_one, self.model_loader.text_encoder_two],
                tokenizers=None,
                prompt=None,
                text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
            )
            
            # Predict noise residual
            unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}
            model_pred = self.model_loader.unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]
            
            # Get target for loss
            if self.config.prediction_type is not None:
                self.model_loader.noise_scheduler.register_to_config(prediction_type=self.config.prediction_type)
            
            if self.model_loader.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.model_loader.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.model_loader.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.model_loader.noise_scheduler.config.prediction_type}")
            
            # Compute loss
            loss = self._compute_loss(model_pred, target, timesteps)
            
            # Debug loss logging
            if self.config.debug_loss and "filenames" in batch:
                for fname in batch["filenames"]:
                    self.accelerator.log({"loss_for_" + fname: loss}, step=self.global_step)
            
            # Gather losses for logging
            avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
            train_loss = avg_loss.item() / self.config.gradient_accumulation_steps
            
            # Backpropagation
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_optimize = self.model_loader.get_trainable_parameters()
                self.accelerator.clip_grad_norm_(params_to_optimize, self.config.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            return train_loss, loss.detach().item()
    
    def _run_validation(self, epoch):
        """Run validation if configured."""
        if self.config.validation_prompt is None or epoch % self.config.validation_epochs != 0:
            return
        
        logger.info(f"Running validation at epoch {epoch}")
        
        # Create pipeline for validation
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.model_loader.vae,
            text_encoder=ModelLoader.unwrap_model(self.model_loader.text_encoder_one, self.accelerator),
            text_encoder_2=ModelLoader.unwrap_model(self.model_loader.text_encoder_two, self.accelerator),
            unet=ModelLoader.unwrap_model(self.model_loader.unet, self.accelerator),
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=self.model_loader.weight_dtype,
        )
        
        # Run validation
        log_validation(pipeline, self.config, self.accelerator, epoch)
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache()
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Initialize trackers
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("text2image-fine-tune", config=vars(self.config))
        
        total_batch_size = (
            self.config.train_batch_size * 
            self.accelerator.num_processes * 
            self.config.gradient_accumulation_steps
        )
        
        logger.info("***** Training Summary *****")
        logger.info(f"  Num examples = {len(self.data_handler.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        
        # Handle resume from checkpoint
        self._handle_resume_from_checkpoint()
        
        # Setup progress bar
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        # Training loop
        for epoch in range(self.first_epoch, self.config.num_train_epochs):
            self.model_loader.unet.train()
            if self.config.train_text_encoder:
                self.model_loader.text_encoder_one.train()
                self.model_loader.text_encoder_two.train()
            
            train_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                step_train_loss, step_loss = self._training_step(batch)
                train_loss += step_train_loss
                
                # Check if optimization step was performed
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0
                    
                    # Handle checkpointing
                    self._handle_checkpointing()
                
                # Update progress bar
                logs = {"step_loss": step_loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if self.global_step >= self.config.max_train_steps:
                    break
            
            # Run validation
            if self.accelerator.is_main_process:
                self._run_validation(epoch)
        
        # Save final model
        self._save_final_model(epoch)
        
        # End training
        self.accelerator.end_training()
        logger.info("Training completed!")
    
    def _handle_resume_from_checkpoint(self):
        """Handle resuming from checkpoint."""
        if self.config.resume_from_checkpoint is None:
            return
        
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.config.resume_from_checkpoint}' does not exist. Starting new training."
            )
            self.config.resume_from_checkpoint = None
            self.global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.config.output_dir, path))
            self.global_step = int(path.split("-")[1])
            
            num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
            self.first_epoch = self.global_step // num_update_steps_per_epoch
    
    def _save_final_model(self, epoch):
        """Save the final trained model."""
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        
        logger.info("Saving final model...")
        
        # Get LoRA layers
        unet = ModelLoader.unwrap_model(self.model_loader.unet, self.accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        
        if self.config.train_text_encoder:
            text_encoder_one = ModelLoader.unwrap_model(self.model_loader.text_encoder_one, self.accelerator)
            text_encoder_two = ModelLoader.unwrap_model(self.model_loader.text_encoder_two, self.accelerator)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
        
        # Save LoRA weights
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=self.config.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        
        # Cleanup
        del unet
        if self.config.train_text_encoder:
            del text_encoder_one
            del text_encoder_two
            del text_encoder_lora_layers
            del text_encoder_2_lora_layers
        torch.cuda.empty_cache()
        
        # Final validation
        if self.config.validation_prompt and self.config.num_validation_images > 0:
            self._run_final_validation(epoch)
    
    def _run_final_validation(self, epoch):
        """Run final validation with the trained model."""
        logger.info("Running final validation...")
        
        # Make sure VAE dtype is consistent
        if self.accelerator.mixed_precision == "fp16":
            self.model_loader.vae.to(self.model_loader.weight_dtype)
        
        # Load pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.model_loader.vae,
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=self.model_loader.weight_dtype,
        )
        
        # Load LoRA weights
        pipeline.load_lora_weights(self.config.output_dir)
        
        # Run inference
        images = log_validation(pipeline, self.config, self.accelerator, epoch, is_final_validation=True)
        logger.info(f"Generated {len(images)} images with prompt: {self.config.validation_prompt}")
        
        del pipeline
        torch.cuda.empty_cache()