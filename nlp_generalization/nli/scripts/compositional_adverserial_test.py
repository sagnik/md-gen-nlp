"""
nli models
"""
import os.path
import re
import evaluate
import numpy as np
import argparse
import transformers
from transformers import DataCollatorWithPadding, TrainingArguments, EvalPrediction
import sys; sys.path.append('/data/shire/projects/generalization/')
import nlp_generalization.yamlenv as yamlenv
from typing import Callable, Dict
from nlp_generalization.utils.nli.dataset_loaders import DATASET_LOADER_MAP
from nlp_generalization.utils.nli.model_loaders import MODEL_LOADER_MAP
from nlp_generalization.reporting import WandbLogger
from adapters import AdapterTrainer


ID_2_LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
LABEL_2_ID = {v: k for k, v in ID_2_LABEL.items()}


def filter_by_length(example, tokenizer):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    input_len = len(tokenizer(premise, hypothesis, truncation=False).input_ids)
    return input_len < tokenizer.model_max_length


def filter_by_num_labels(example):
    return example['label'] in list(range(len(ID_2_LABEL)))


def preprocess_nli_batch(examples):
    premises = examples["premise"]
    hypotheses = examples["hypothesis"]

    def generate_input(_premise, _hypothesis):
        return " ".join(["premise:", _premise, "hypothesis:", _hypothesis])

    return [generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]


def tokenize_function(examples, tokenizer, model: str):
    if model.startswith("t5") or model.startswith("opt"):
        return tokenizer(preprocess_nli_batch(examples))
    return tokenizer(examples['premise'], examples['hypothesis'])


def compute_metrics(eval_pred: EvalPrediction):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    # for t5, this will actually return a tuple so handle that
    if type(logits) == tuple:
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_nli_seq_cls(tokenizer, trainer, dataset_loader_fn: Callable, model: str):
    """
    evaluation when nli is modeled as a sequence classification task
    :return:
    """
    test_data = dataset_loader_fn()
    test_data_filtered = test_data.filter(filter_by_length, fn_kwargs={"tokenizer": tokenizer})
    test_data_filtered = test_data_filtered.filter(filter_by_num_labels)
    print(f"original: {len(test_data)}, filtered: {len(test_data_filtered)}, "
          f"{len(test_data) - len(test_data_filtered)} data points filtered out")
    tokenized_test_data = test_data_filtered.map(tokenize_function, batched=True,
                                                 fn_kwargs={"tokenizer": tokenizer, "model": model})
    
    results = trainer.evaluate(tokenized_test_data)
    return results


def change_config_dict(_config: Dict) -> Dict:
    model = _config["model"]["model_name"]
    if "bert-" in model:
        _config["base_model"] = "bert"
    elif "roberta-" in model:
        _config["base_model"] = "roberta"
    elif "t5" in model:
        _config["base_model"] = "t5"
    elif "opt" in model:
        _config["base_model"] = "opt"
    else:
        raise RuntimeError(f"no known base model for {model}")

    if "base-" in model:
        _config["size"] = "base"
    elif "large-" in model:
        _config["size"] = "large"
    elif "1.3b" in model:
        _config["size"] = "large"
    elif "350m" in model:
        _config["size"] = "base"
    else:
        raise RuntimeError(f"no known size for {model}")

    pattern = "{0}-\d+.*\d+[M|K]"
    for trainer in ["lora", "adapter"]:
        if f"{trainer}-" in model:
            _config["training"] = re.findall(pattern.format(trainer), model)[0]
    else:
        _config["training"] = "full-training"
    return _config


def load_trainer(model, tokenizer, _compute_metrics: Callable, model_type: str, **kwargs):
    """
    :param model:
    :param tokenizer:
    :param _compute_metrics
    :param model_type
    :return:
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if kwargs.get("large_model", False):
        training_args = TrainingArguments(output_dir="NA", do_train=False, do_eval=True, report_to=None) # stop default wandb logging
    else:
        training_args = TrainingArguments(output_dir="NA", do_train=False, do_eval=True, report_to=None,
                                          per_device_eval_batch_size=2)
    if model_type == "adapter":
        return AdapterTrainer(
            model=model,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics,
            args=training_args
        )
    return transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        args=training_args
    )


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/generated-configs/snli-bt-bertbc-plain-1.yml")
    args = parser.parse_args()

    config = yamlenv.load(open(args.config))
    
    # configure logging
    exp_name = os.path.split(os.path.splitext(args.config)[0])[-1]
    
    logger = WandbLogger(exp_name=exp_name, exp_config=change_config_dict(config))
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"])
    # model
    model_loader_fn = MODEL_LOADER_MAP[config["model"]["model_type"]]
    model = model_loader_fn(**config["model"])
    # trainer
    large_model = "large" in config["model"]["base_model"] or "1.3b" in config["model"]["base_model"]
    trainer = load_trainer(model=model, tokenizer=tokenizer, _compute_metrics=compute_metrics,
                           model_type=config["model"]["model_type"], large_model=large_model)
    # data
    dataset_loader_fn = DATASET_LOADER_MAP[config["dataset"]]
    result = evaluate_nli_seq_cls(tokenizer=tokenizer, trainer=trainer, dataset_loader_fn=dataset_loader_fn,
                                  model=config["model"]["base_model"])
    
    
    logger.done(result)


if __name__ == "__main__":
    main()
