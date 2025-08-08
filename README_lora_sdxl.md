# Fine-tuning Stable Diffusion XL with LoRA

This project provides a complete pipeline for training **LoRA adapters** for **Stable Diffusion XL (SDXL)** using AWS SageMaker and modular Python components.

LoRA—short for *Low-Rank Adaptation*—is a clever way to fine-tune large models without updating all their weights. It was introduced by Microsoft in [this paper](https://huggingface.co/papers/2106.09685) and works by injecting small trainable layers (low-rank matrices) into a frozen pretrained model. This approach:

- Keeps the original model safe from catastrophic forgetting  
- Greatly reduces the number of trainable parameters  
- Makes fine-tuned weights lightweight and easy to share  
- Adds a `scale` knob to control how strongly your LoRA influences generation

---

## Project Structure

```
├── infrastructure/           # AWS CDK infrastructure code
│   ├── lib/
│   │   ├── constructs/      # Reusable CDK constructs
│   │   └── infrastructure-stack.ts
│   └── package.json
└── src/
    └── containers/
        └── start_tuning/    # Training container
            ├── Dockerfile
            ├── build_pipeline.py      # SageMaker pipeline orchestration
            ├── modules/
            │   └── tuning_step.py     # Training step definition
            ├── src/                   # Core training modules
            │   ├── config.py          # Configuration management
            │   ├── data_handler.py    # Dataset loading and preprocessing
            │   ├── model_loader.py    # Model and tokenizer loading
            │   ├── train.py           # Main training script
            │   ├── trainer.py         # Training loop implementation
            │   └── utils_train.py     # Training utilities
            └── utils/
                └── session_utils.py   # AWS session management
```

---

## Key Features

### Modular Architecture
- **Configuration Management**: Comprehensive config system with validation (`config.py`)
- **Model Loading**: Separate model and tokenizer loading (`model_loader.py`)
- **Data Handling**: Dedicated dataset processing (`data_handler.py`)
- **Training Logic**: Isolated trainer class (`trainer.py`)

### AWS SageMaker Integration
- **Pipeline Orchestration**: Automated SageMaker pipeline creation
- **Scalable Training**: Configurable instance types and counts
- **S3 Integration**: Automatic model and dataset storage

### Advanced Training Features
- **Mixed Precision**: Support for fp16/bf16 training
- **Memory Optimization**: Gradient checkpointing and xformers support
- **Validation**: Built-in validation image generation
- **Checkpointing**: Automatic checkpoint saving and resumption
- **Monitoring**: TensorBoard and Weights & Biases integration

---

## Getting Started

### Prerequisites
- AWS account with appropriate permissions
- Docker
- Node.js and npm (for CDK)
- Python 3.8+

### Environment Setup
Set required environment variables:
```bash
export AWS_REGION=us-east-1
export BUCKET_NAME=your-s3-bucket
```

### Training Parameters

Key configuration options (see `config.py` for full list):

**Model Parameters**:
- `--pretrained_model_name_or_path`: Base SDXL model (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `--pretrained_vae_model_name_or_path`: VAE model (default: `madebyollin/sdxl-vae-fp16-fix`)

**Dataset Parameters**:
- `--dataset_name`: HuggingFace dataset (default: `lambdalabs/naruto-blip-captions`)
- `--train_data_dir`: Local training data directory
- `--resolution`: Input image resolution (default: 1024)

**Training Parameters**:
- `--learning_rate`: Learning rate (default: 1e-4)
- `--train_batch_size`: Batch size (default: 16)
- `--num_train_epochs`: Training epochs (default: 100)
- `--rank`: LoRA rank (default: 4)

**Memory Optimization**:
- `--mixed_precision`: Use fp16/bf16 (default: None)
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--enable_xformers_memory_efficient_attention`: Use xformers

### Running Training

#### Option 1: SageMaker Pipeline (Recommended)
```bash
cd src/containers/start_tuning
python build_pipeline.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --dataset_name lambdalabs/naruto-blip-captions \
    --validation_prompt "cute dragon creature"
```

#### Option 2: Direct Training
```bash
cd src/containers/start_tuning/src
python train.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --dataset_name lambdalabs/naruto-blip-captions \
    --output_dir ./output \
    --resolution 1024 \
    --learning_rate 1e-4 \
    --rank 4 \
    --train_batch_size 4 \
    --num_train_epochs 10 \
    --validation_prompt "a cute dragon creature" \
    --mixed_precision fp16
```

---

## Configuration

The training script supports extensive configuration through command-line arguments. Key categories:

- **Model & Dataset**: Model paths, dataset configuration
- **Training**: Learning rate, batch size, epochs, mixed precision
- **LoRA**: Rank, target modules
- **Optimization**: Adam parameters, gradient clipping
- **Memory**: Xformers, gradient checkpointing
- **Validation**: Validation prompts and frequency
- **Logging**: TensorBoard, Weights & Biases integration

See `src/containers/start_tuning/src/config.py` for the complete list of parameters.

---

## Infrastructure

The project includes AWS CDK infrastructure for:
- **VPC**: Default VPC configuration
- **ECR**: Container registry for training images
- **ECS**: Container orchestration
- **S3**: Model and dataset storage

Deploy with:
```bash
cd infrastructure
npm install
npx cdk deploy
```

---

## Monitoring

**TensorBoard**: Default logging backend
```bash
tensorboard --logdir ./output/logs
```

**Weights & Biases**: Set `--report_to wandb`
- Automatic sample generation during validation
- Loss tracking and metrics visualization

> **Note**: SDXL's original VAE can be unstable. The default configuration uses the [fp16-fix VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) for better numerical stability.

---

## Troubleshooting

**Memory Issues**:
- Reduce `--train_batch_size`
- Enable `--gradient_checkpointing`
- Use `--mixed_precision fp16`
- Enable `--enable_xformers_memory_efficient_attention`

**MPS + bfloat16 Incompatibility**:
- Use `--mixed_precision fp16` instead of `bf16` on Apple Silicon

**Dataset Loading**:
- Ensure proper dataset format (image + text columns)
- Check dataset name mapping in `build_pipeline.py`