import fire
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer

import adapters
from adapters import AdapterTrainer

from peft import LoraConfig, TaskType, get_peft_model

from tqdm.auto import tqdm
import evaluate
import numpy as np
import collections
import random

import os
from datetime import datetime

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def build_dataset(tokenizer):
    
    def filter_function(examples):
        question = examples["question"]
        context = examples["context"]
        input_len = len(tokenizer(question,context, truncation=False).input_ids)
        return (input_len < tokenizer.model_max_length)
    
    def tokenize_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    dataset = load_dataset("squad", split="train")
    dataset = dataset.filter(filter_function)
    train_dataset = dataset.map(tokenize_train, batched=True, num_proc=4, remove_columns=dataset.column_names)
    return train_dataset


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    
    metric = evaluate.load("squad")
    
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
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
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def evaluate_test_sets(tokenizer, trainer, model_name):
    def newsqa2squad(sample):
        answers_text = sample["answers"]
        answers_start = sample["labels"][0]["start"]
        sample["answers"] = {'text':answers_text, 'answer_start':answers_start}
        return sample
    
    def filter_out_yes_no_answers(sample):
        return sample['answer'].lower() != 'yes' and sample['answer'].lower() != 'no'

    def advhotpotqa2squad(sample):
        supporting_facts = sample['supporting_facts']
        context_texts = sample['context']
        titles_to_sentences = collections.defaultdict(list)

        for title,sentence in zip(context_texts['title'],context_texts['sentences']):
            for s in sentence:
                text = s.strip()
                titles_to_sentences[title].append(text)

        sf_titles = supporting_facts['title']
        sf_sent_id = supporting_facts['sent_id']

        sf_context=[]
        for i,sf_title in enumerate(sf_titles):
            try:
                text = titles_to_sentences[sf_title][sf_sent_id[i]]
            except:
                sample['_id'] = 'skip'
            sf_context.append(text.strip())

        sample['id'] = sample['_id']
        # ---Shuffle sentences in context and join them into a single string ---#
        context_list = sf_context + titles_to_sentences['added']
        random.shuffle(context_list)
        sample['context'] = ' '.join(context_list)
        sample['answers'] = {'text':[sample['answer']], 'answer_start':[sample['context'].find(sample['answer'])]}
        return sample

    def filter_function(examples):
        question = examples["question"]
        context = examples["context"]
        input_len = len(tokenizer(question,context, truncation=False).input_ids)
        return (input_len < tokenizer.model_max_length)

    def tokenize(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding="max_length",
        )
        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]
        inputs["example_id"] = example_ids
        return inputs
    
    test_datasets = ['squad','lucadiliello/newsqa','sagnikrayc/adversarial_hotpotqa'] #
    dataset2split = {'squad':"validation", 'lucadiliello/newsqa':"validation", 'sagnikrayc/adversarial_hotpotqa':"validation"}
    res = []
    metric = evaluate.load("squad")
    
    for dataset_str in test_datasets:
        target_split = dataset2split[dataset_str] 
        dataset = load_dataset(dataset_str, split=target_split)
        
        if dataset_str == 'lucadiliello/newsqa': dataset = dataset.map(newsqa2squad, num_proc=4).rename_column("key","id")
        
        if dataset_str == 'sagnikrayc/adversarial_hotpotqa':
            dataset = dataset.filter(filter_out_yes_no_answers, num_proc=4) 
            dataset = dataset.map(advhotpotqa2squad, num_proc=4, remove_columns=["type",'level','supporting_facts','answer','_id']).filter(lambda x: x['id']!='skip')

        dataset = dataset.filter(filter_function)
        
        tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4,remove_columns=dataset.column_names)

        predictions, _, _ = trainer.predict(tokenized_dataset)
        start_logits, end_logits = predictions
        results = compute_metrics(start_logits, end_logits, tokenized_dataset, dataset, metric)
        res.append([model_name, dataset_str,results['f1']])
    return res

def log_and_save_results(res,
    results_dir = '../../result_logs',
    outfile_name = 'squad_finetuning_performances.csv'
):
    outfile_path = os.path.join(results_dir, outfile_name)

    if not os.path.exists(results_dir): os.mkdir(results_dir)

    if not os.path.exists(outfile_path):
        with open(outfile_path,'a', newline='\n') as f:
            f.write("date_time; model_name; dataset; f1_score\n")

    today = datetime.today()

    for i  in res:
        model_name, dataset_str, accuracy = i
        with open(outfile_path,'a', newline='\n') as f:
            f.write(f"{today}; {model_name}; {dataset_str}; {accuracy}\n")
        print(f"F1 score of {model_name} on {dataset_str} dataset: {accuracy}")

def main(
    model_checkpoint,
    seed: int=42,
    batch_size: int=64,
    num_train_epochs: int=3,
    num_proc: int=4,
    output_dir: str="../../result_logs",
    do_train: bool=True,
    do_eval: bool=True,
    do_log: bool=True,
    use_adapter: bool=False,
    use_peft: bool=False,
    save_path: str="/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/nlp-gen/qa" 
):
    
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    checkpoint = model_checkpoint
    model_name = checkpoint.split("/")[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    
    train_dataset = build_dataset(tokenizer)
    
    data_collator = DefaultDataCollator()
    
    # LOAD ADAPTER
    if use_adapter:
        print("Initializing Adapters for transformer model")
        adapters.init(model)
        model.add_adapter("squad", config="seq_bn")
        model.train_adapter("squad")
        # print number of trainable parameters
        summary = model.adapter_summary(as_dict=True)
        print(f"trainable params: {summary[0]['#param']:,d} || all params: {summary[1]['#param']:,d} || trainable%: {summary[0]['%param']}")
        # edit model name
        num_param = human_format(summary[0]['#param'])
        model_name = f"ADAPTER/{model_name}-bn-adapter-{num_param}"
    
    # LOAD PEFT MODEL
    if use_peft:
        print("Loading PEFT(LORA) Model")
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        # print number of trainable paramaeters
        model.print_trainable_parameters()
        # edit model name
        num_param = human_format(model.get_nb_trainable_parameters()[0])
        model_name = f"PEFT/{model_name}-lora-{num_param}"
    
    save_path = f"{save_path}/{model_name}-squad"
    
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="no",
        save_strategy = "epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        fp16=True,
        #push_to_hub=True,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    if do_train:
        trainer.train()
    
    # EVALUATE PERFORMANCE ON TEST SETS   
    if do_eval:
        results = evaluate_test_sets(tokenizer, trainer, model_name)
        # LOG RESULT METRICS
        if do_log:
            log_and_save_results(results, results_dir = output_dir, outfile_name = 'squad_finetuning_performances.csv')
        else:
            print(results)
    

if __name__ == "__main__":
    fire.Fire(main)