import os
import argparse
from utils.session_utils import create_boto_and_sagemaker_sessions, get_role
from modules.tuning_step import get_training_step
from sagemaker.workflow.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Build and start SageMaker pipeline for finetuning SDXL LoRA.")
    
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="The path or identifier of the pretrained base model."
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path", 
        type=str, 
        default="madebyollin/sdxl-vae-fp16-fix",
        help="The path or identifier of the pretrained VAE model."
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="lambdalabs/naruto-blip-captions",
        help="The dataset name to use for training."
    )
    parser.add_argument(
        "--validation_prompt", 
        type=str, 
        default="cute dragon creature",
        help="A prompt used to generate validation images."
    )
    parser.add_argument(
        "--dataset_name_mapping", 
        type=str, 
        default='{"lambdalabs/naruto-blip-captions": ["image", "text"]}',
        help="JSON string or path to JSON file mapping dataset name to (image_column, caption_column)."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    region = os.environ['AWS_REGION']
    bucket = os.environ['BUCKET_NAME']

    base_uri = f"s3://{bucket}/train/text-to-image-sdxl-lora-finetune"

    boto_session, sess = create_boto_and_sagemaker_sessions(region)
    role = get_role(boto_session)


    # Steps
    tuning_step, train_inst_type, train_inst_count = get_training_step(
        role, sess, base_uri, args.pretrained_model_name_or_path, args.pretrained_vae_model_name_or_path, args.dataset_name, args.validation_prompt, args.DATASET_NAME_MAPPING 
    )


    # Pipeline
    pipeline = Pipeline(
        name=f"finetune-text-to-image-naruto",
        parameters=[
            train_inst_type,
            train_inst_count,
        ],
        steps=[get_training_step]
    )

    pipeline.upsert(role_arn=role)
    pipeline.start(parameters={
        "TrainingInstanceType": "ml.g5.4xlarge",
        "TrainingInstanceCount": 1
    })
    
if __name__ == "__main__":
    main()