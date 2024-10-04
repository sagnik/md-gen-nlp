"""
Script for training and pushing models to the Hugging Face Model Hub.

This script trains multiple model instances using a specified dataset and configuration parameters, and then pushes the trained models to the Hugging Face Model Hub.

Usage:
    python train.py --model_checkpoint "PATH/to/HF/model"[optional_arguments] 

Args:
    model_checkpoint: Path to pretrained model or model identifier from huggingface.co/models

Example usage:
    python train.py --model_checkpoint "bert-base-cased"
"""

import sys
import random
from typing import Optional, Callable
from dataclasses import dataclass, field

from huggingface_hub import login

from transformers import AutoTokenizer
from transformers import HfArgumentParser

from nlp_generalization.configs.mrc.dataset_config import DatasetArguments
from nlp_generalization.configs.mrc.model_config import ModelArguments
from nlp_generalization.configs.mrc.train_config import TrainConfig

from nlp_generalization.utils.mrc.dataset_utils import get_preprocessed_dataset
from nlp_generalization.utils.mrc.train_utils import *

def main():
    """
    Main function for training and pushing models to the Hugging Face Model Hub.
    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, dataset_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, train_args = parser.parse_args_into_dataclasses()
        
    # Update configs
    model_args.model_name = get_num_trainable_parameters(train_args, model_args)
    train_args.checkpoints_save_path = f"{train_args.checkpoints_save_path}/MULTI-MODEL/{dataset_args.dataset_name}/{train_args.training_stratergy}"

    # Login to HF Hub
    login(token=train_args.hf_write_token)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, model_max_length=dataset_args.max_seq_length)

    # Build Dataset
    train_dataset = get_preprocessed_dataset(tokenizer, dataset_args, model_args, split="train")

    if dataset_args.max_train_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        max_train_samples = min(len(train_dataset), dataset_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    # Define Model Initialization Function
    def model_init():
        """
        Function to initialize the model.
        """
        model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_checkpoint)      
        # LOAD BN ADAPTER MODEL
        if train_args.use_adapter:
            adapters.init(model)
            model.add_adapter(f"{dataset_args.dataset_name}", config="seq_bn")
            model.train_adapter(f"{dataset_args.dataset_name}")
        # LOAD PEFT MODEL
        if train_args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05
            )
            model = get_peft_model(model, lora_config)
        return model
    
    seeds_used = []
    
    # Train model instances
    for model_num in range(train_args.num_models_to_train): 
        # Set Seed
        if train_args.use_random_seed:
            seed = random.randint(1,100)
            while seed in seeds_used:
                seed = random.randint(1,100)
            seeds_used.append(seed)
            train_args.seed = seed

        # Shuffle Training Data
        if train_args.shuffle_train_data:
            train_dataset = train_dataset.shuffle(seed=train_args.seed)

        # Update train_args
        train_args.checkpoint_dir = f"{train_args.checkpoints_save_path}/{model_args.model_name}/{model_args.model_name}-{dataset_args.dataset_name}-model{model_num+1}"
        train_args.hf_hub_id = f"{train_args.hf_org_name}/{model_args.model_name}-{dataset_args.dataset_name}-model{model_num+1}"

        print(f"Model Num: {model_num+1}")
        print(f"Seed: {train_args.seed}")

        # Load Trainer
        trainer = load_trainer(model_init, tokenizer, train_dataset, train_args)

        # Train Model
        trainer.train() 

        # PUSH MODEL TO HUB
        if train_args.push_to_hub:
            push_model_to_hub(trainer, train_args, dataset_args)
    
if __name__ == "__main__":
    main()