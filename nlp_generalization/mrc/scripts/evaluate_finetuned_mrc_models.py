"""
mrc models
"""
import json
import pandas as pd
import os.path
import re
import evaluate
import numpy as np
import argparse
import sys; sys.path.append('/data/shire/projects/generalization/')
from datasets import Dataset
import transformers
from transformers import TrainingArguments
import nlp_generalization.yamlenv as yamlenv
from typing import Callable, Dict
from nlp_generalization.utils.mrc.dataset_loaders import DATASET_LOADER_MAP
from nlp_generalization.utils.mrc.model_loaders import MODEL_LOADER_MAP
from nlp_generalization.reporting import WandbLogger
from adapters import AdapterTrainer



def filter_by_length(examples, tokenizer, model: str):
    question = examples["question"]
    context = examples["context"]
    if model.startswith("bert") or model.startswith("roberta"):
        input_len = len(tokenizer(question,context, truncation=False).input_ids)
        return (input_len < tokenizer.model_max_length)
    else:
        def generate_question_context_pairs(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        input_len = len(tokenizer(generate_question_context_pairs(question, context), truncation=False).input_ids)
        return (input_len < 512) # OPT models dont have model_max_length defined, so I fix it to 512

def render_and_write_json_file(json_fpath, theoretical_answers, predicted_answers):
    dictObj = {}

    # Merge theoretical answers and predicted answers into a single dict
    theoretical_answers_df = pd.DataFrame(theoretical_answers)
    predicted_answers_df = pd.DataFrame(predicted_answers)
    merged_df = pd.merge(theoretical_answers_df, predicted_answers_df, on='id')
    merged_df.set_index('id', inplace= True)
    new_dictObj = merged_df.to_dict("index")
    
    #DEBUG
    #print(dictObj.keys())
    #print(new_dictObj.keys())
    
    # Check if file exists
    if os.path.isfile(json_fpath) is True:
        with open(json_fpath) as f:
            dictObj = json.load(f)
      
    # Update dictObj
    dictObj.update(new_dictObj)

    # Write/Dump updated obj to json file
    with open(json_fpath, 'w') as json_file:
        json.dump(dictObj, json_file, 
                            indent=4,  
                            separators=(',',':'))
    
    print(f"Updated {json_fpath}")
    
def compute_metrics(start_logits, end_logits, features, examples, metric, model, json_fpath):
    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    
    for feature_index,example in enumerate(examples):
        example_id = example["id"]
        
        if model["base_model"].startswith("bert") or model["base_model"].startswith("roberta"):
            context = example["context"]
        elif model["base_model"].startswith("t5") or model["base_model"].startswith("opt"):
            context = f"question: {example['question'].lstrip()} context: {example['context'].lstrip()}"
        
        answers = []

        # Loop through all features associated with that example
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = features[feature_index]["offset_mapping"]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers with a length that is either < 0 or > max_answer_length
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                    or end_index >= len(offsets)
                ):
                    continue
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue

                answer = {
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                }
                answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    label_text = [{"id": ex["id"], "label_text": ex["answers"]["text"]} for ex in examples]
    # Write predictions and labels to a json file
    render_and_write_json_file(json_fpath, label_text, predicted_answers)

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def tokenize_function(examples, tokenizer, model: str):
    if model.startswith("t5") or model.startswith("opt"):
        questions = examples["question"]
        contexts = examples["context"]

        def generate_question_context_pairs(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        question_context_pairs = [generate_question_context_pairs(question, context) for question, context in zip(questions, contexts)]

        inputs = tokenizer(
            question_context_pairs,
            return_offsets_mapping=True,
            padding=True, 
            truncation=True,
        )

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 0 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = examples["id"]

        return inputs
    # Else tokenize for bert and roberta models
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    inputs = tokenizer(
        questions,
        contexts,
        return_offsets_mapping=True,
    )

    for i in range(len(inputs["input_ids"])):
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = examples["id"]

    return inputs

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


def load_trainer(model, tokenizer, model_type: str, **kwargs):
    """
    :param model:
    :param tokenizer:
    :param _compute_metrics
    :param model_type
    :return:
    """
    #print(f"\nLarge Model Flag: {kwargs.get('large_model', False)}\n")
    if kwargs.get("large_model", False):
        training_args = TrainingArguments(output_dir="NA", do_train=False, do_eval=True, report_to=None,
                                        per_device_eval_batch_size=1, eval_accumulation_steps=256) # stop default wandb logging
    else:
        training_args = TrainingArguments(output_dir="NA", do_train=False, do_eval=True, report_to=None,
                                          per_device_eval_batch_size=1, eval_accumulation_steps=256)
    if model_type == "adapter":
        return AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
        )
    return transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args
    )

'''
To-DO
remove filter_by_length?
'''
def evaluate_mrc(tokenizer, trainer, dataset_loader_fn: Callable, model, json_fpath:str):
    '''
    Evaluation for MRC task
    '''
    metric = evaluate.load("squad")
    test_data = dataset_loader_fn()
    test_data_filtered = test_data.filter(filter_by_length, fn_kwargs={"tokenizer": tokenizer, "model": model["base_model"]})
    print(f"original: {len(test_data)}, filtered: {len(test_data_filtered)}, "
          f"{len(test_data) - len(test_data_filtered)} data points filtered out")
    
    
    # for i in range(len(test_data_filtered)):
    #     print(f'Done for {i} out of {len(test_data_filtered)}')
    #     curr_elem = test_data_filtered[i]
        
    #     curr_elem = Dataset.from_dict({'context': curr_elem['context'], 'question': curr_elem['question'],\
    #         'answers': curr_elem['answers'], 'id': curr_elem['id'], 'labels': curr_elem['labels']})
    
    
    tokenized_test_data = test_data_filtered.map(tokenize_function, batched=True,
                                                fn_kwargs={"tokenizer": tokenizer, "model": model["base_model"]})

    
    # tokenized_test_data = tokenized_test_data.remove_columns(["context", "question", "answers", "id", "example_id", "labels"])
    
    predictions, _, _ = trainer.predict(tokenized_test_data)
    start_logits, end_logits = predictions[0], predictions[1]
    results = compute_metrics(start_logits, end_logits, tokenized_test_data, test_data_filtered, metric, model, json_fpath)
    
    return {'mrc_eval_f1': results['f1']}

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/squad-bertbc.yml")
    args = parser.parse_args()

    config_dir = "configs/generated-configs/"
    
    # for config_name in sorted(os.listdir(config_dir)):
    #     if 't5' in config_name and 'adapter' in config_name and 'newsqa' in config_name:

    #         config   = yamlenv.load(open(f'{config_dir}/{config_name}'))
    #         print(f"Config: {config_name}")
    
    config = yamlenv.load(open(args.config))
    # configure logging
    exp_name = os.path.split(os.path.splitext(args.config)[0])[-1]
    
    # exp_name = config_name
    # create output dir for json files
    json_dir = f"json"
    os.makedirs(json_dir, exist_ok=True)
    json_fpath = os.path.join(json_dir, exp_name)+".json"
    
    print(f"Json file path: {json_fpath}\n\n\n")
    
    # logger = WandbLogger(exp_name=exp_name, exp_config=change_config_dict(config))
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"])
    # model
    model_loader_fn = MODEL_LOADER_MAP[config["model"]["model_type"]]
    model = model_loader_fn(**config["model"])
    # trainer
    large_model = "large" in config["model"]["base_model"] or "1.3b" in config["model"]["base_model"]
    trainer = load_trainer(model=model, tokenizer=tokenizer,
                        model_type=config["model"]["model_type"], large_model=large_model)
    # data
    dataset_loader_fn = DATASET_LOADER_MAP[config["dataset"]]
    result = evaluate_mrc(tokenizer=tokenizer, trainer=trainer, dataset_loader_fn=dataset_loader_fn,
                                model=config["model"], json_fpath=json_fpath)
    
    print(result)
            # logger.done(result)


if __name__ == "__main__":
    main()