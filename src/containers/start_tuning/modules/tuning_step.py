from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter
)
from sagemaker.workflow.steps import TuningStep
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger
)
import datetime


def get_training_step(
    role,
    sess,
    base_uri,
    pretrained_model_name_or_path,
    pretrained_vae_model_name_or_path,
    dataset_name,
    validation_prompt,
    DATASET_NAME_MAPPING 
):
    # Parameters
    train_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.g5.2xlarge"
    )
    train_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )
    dataset_columns = DATASET_NAME_MAPPING.get(dataset_name.default_value, ("", ""))
    image_column = dataset_columns[0]
    caption_column = dataset_columns[1]
    pretrained_model_name_or_path = ParameterString(name="pretrained_model_name_or_path", default_value=pretrained_model_name_or_path)
    pretrained_vae_model_name_or_path = ParameterString(name="pretrained_vae_model_name_or_path", default_value=None)
    dataset_name = ParameterString(name="dataset_name", default_value=dataset_name)

    output_path = ParameterString(name="OutputPath", default_value=f"{base_uri}/output-naruto-model-lora-sdxl")
    checkpoint_s3_uri = ParameterString(name="CheckpointS3URI", default_value=f"{base_uri}/checkpoints")

    checkpoint_local_path = '/opt/ml/checkpoints'
    base_job_name = f"finetune-text-to-image-{datetime.datetime.now().strftime('%Y%m%d')}"

    # Static hyperparameters
    hyperparameters = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path.default_value,
        "pretrained_vae_model_name_or_path": pretrained_vae_model_name_or_path.default_value,
        "dataset_name": dataset_name.default_value,
        "checkpointing_steps": 500,
        "caption_column": "text", 
        "resolution": 1024, 
        "lr_scheduler": "constant",
        "checkpoint_local_path": checkpoint_local_path,
        "lr_warmup_steps": 0,
        "mixed_precision": "fp16",
        "validation_prompt": validation_prompt,
        "report_to": "wandb",
        "image_column": image_column,
        "caption_column": caption_column
    }

    # Hyperparameter tuning search space
    hyperparameter_ranges = {
        "train_batch_size": IntegerParameter(1, 4),
        "num_train_epochs": IntegerParameter(2, 5),
        "learning_rate": ContinuousParameter(1e-05, 1e-04),
        "adam_weight_decay": ContinuousParameter(1e-02, 1e-01),
    }

    # Metric regex
    metrics_definitions = [
        {"Name": "loss", "Regex": "train_loss: (.*)"}
    ]

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        instance_type=train_instance_type,
        instance_count=train_instance_count,
        role=role,
        framework_version="2.0.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=output_path.default_value,
        checkpoint_s3_uri=checkpoint_s3_uri.default_value,
        checkpoint_local_path=checkpoint_local_path,
        disable_profiler=True,
        base_job_name=base_job_name,
        metric_definitions=metrics_definitions,
        environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},
        sagemaker_session=sess
    )

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="loss",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metrics_definitions,
        max_jobs=4,
        max_parallel_jobs=2,
        objective_type="Minimize",
        base_tuning_job_name=base_job_name,
        early_stopping_type="Auto"
    )

    hpo_args = tuner.fit(
        job_name=base_job_name + "-tuning",
        wait=False
    )

    tuning_step = TuningStep(
        name="HPTuning",
        step_args=hpo_args
    )

    return tuning_step, train_instance_type, train_instance_count
