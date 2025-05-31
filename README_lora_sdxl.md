# Fine-tuning Stable Diffusion XL with LoRA

This project walks you through training **LoRA adapters** for **Stable Diffusion XL (SDXL)**.

LoRA—short for *Low-Rank Adaptation*—is a clever way to fine-tune large models without updating all their weights. It was introduced by Microsoft in [this paper](https://huggingface.co/papers/2106.09685) and works by injecting small trainable layers (low-rank matrices) into a frozen pretrained model. This approach:

- Keeps the original model safe from catastrophic forgetting  
- Greatly reduces the number of trainable parameters  
- Makes fine-tuned weights lightweight and easy to share  
- Adds a `scale` knob to control how strongly your LoRA influences generation

---

## Training

To get started, run the Fargate after deployment. It is possible to overwrite the container environment variables


**Tip**: If you're using [Weights and Biases](https://docs.wandb.ai/quickstart), you'll see samples generated during training—super useful for debugging and progress tracking.

> Note: SDXL’s original VAE can be unstable. Use the [fp16-fix VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) for better results.

---

* [TODO]: Documentation on DeepSeed
* [TODO]: Batch Transform and
* [TODO]: Serverless Inference