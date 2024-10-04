"""
To-Do:

* Update Train_Config to include metric_name. Also remove eval_accumulation_steps

* Add documentation for load_trainer function
* Add documentation for compute_metrics function

* load_trainer - add condition for including validation/train dataset only if available. This should allow us to reuse this function for evaluation script
"""

from typing import Callable

import evaluate
import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction

import adapters
from adapters import AdapterTrainer

from peft import LoraConfig, TaskType, get_peft_model

from nlp_generalization.utils.nli.dataset_utils import NUM_LABELS

DATASET_TAGS_MAP = {
    "snli": "snli",
    "multi_nli": "multi_nli",
    "snli-cf-kaushik": "sagnikrayc/snli-cf-kaushik",
    "snli-bt": "sagnikrayc/snli-bt",
    "sick": "RobZamp/sick"
}

def human_format(num):
    """
    Convert a large number into a human-readable format.

    Args:
        num (float): The number to be formatted.

    Returns:
        str: The formatted number with a magnitude suffix (e.g., K for thousands, M for millions).
    """
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def get_num_trainable_parameters(train_config, model_config):
    """
    Get the number of trainable parameters for the specified model configuration.

    Args:
        train_config (TrainConfig): Configuration for training.
        model_config (ModelConfig): Configuration for the model.

    Returns:
        str: Name of the model along with the number of trainable parameters.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_config.model_checkpoint, num_labels=NUM_LABELS)

    if train_config.use_peft:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        num_param = human_format(model.get_nb_trainable_parameters()[0])
        model_name = f"{model_config.model_name}-lora-{num_param}"
        
    elif train_config.use_adapter:
        adapters.init(model)
        model.add_adapter("tmp", config="seq_bn")
        model.train_adapter("tmp")
        summary = model.adapter_summary(as_dict=True)
        num_param = human_format(summary[0]['#param'])
        model_name = f"{model_config.model_name}-bn-adapter-{num_param}"
        
    else:
        model_name = model_config.model_name
    
    del(model)
    return model_name

def compute_metrics(eval_pred: EvalPrediction):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    # for t5, this will actually return a tuple so handle that
    if type(logits) == tuple:
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def load_trainer(model_init_fn: Callable, tokenizer, data_collator, train_dataset, validation_dataset, train_config):
    """
    Load trainer for training a model.

    Args:

    Returns:
        Trainer: Trainer object for training the model.
    """
    training_args = TrainingArguments(
        output_dir=train_config.checkpoint_dir, 
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        save_total_limit=1,
        load_best_model_at_end=True, 
        metric_for_best_model=train_config.metric_name, 
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.train_batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        weight_decay=train_config.weight_decay,
        overwrite_output_dir=train_config.overwrite_output_dir,
        seed=train_config.seed,
        push_to_hub=False,#train_config.push_to_hub,
        hub_model_id=train_config.hf_hub_id, 
        report_to=None
    )
    
    if train_config.use_adapter:
        return AdapterTrainer(
            model_init=model_init_fn,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    return Trainer(
        model_init=model_init_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

def push_model_to_hub(trainer, train_config, dataset_config):
    """
    Pushes the trained model and associated artifacts to the Hugging Face Model Hub.

    Args:
        trainer (Trainer): Trainer object used for training the model.
        train_config (TrainConfig): Configuration for training.
        dataset_config (DatasetConfig): Configuration for the dataset.

    Returns:
        None
    """
    kwargs = {"dataset": dataset_config.dataset_name, "dataset_tags": DATASET_TAGS_MAP[dataset_config.dataset_name]}

    if train_config.use_adapter:
        trainer.model.push_adapter_to_hub(
            repo_name=train_config.hf_hub_id,
            adapter_name=f"{dataset_config.dataset_name}",
            datasets_tag=f"{dataset_config.dataset_name}",
            token=train_config.hf_write_token,
        )
        trainer.push_to_hub(**kwargs)

    elif train_config.use_peft:
        trainer.model = trainer.model.merge_and_unload()
        trainer.push_to_hub(**kwargs)

    else:
        trainer.push_to_hub(**kwargs)