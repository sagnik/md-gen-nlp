import transformers
import json
import datasets
import os
import evaluate
import numpy as np
import random
import argparse

SETI_TASK_1_TRAIN_NAT = f"{os.environ['GENERALIZATION_HOME']}" \
                 f"/data/compositionality/nli/SETI/1.task1-primitive-composition/task1.2/composition/" \
                        f"train+val_PI_nat.json"
SETI_TASK_1_TRAIN_VER = f"{os.environ['GENERALIZATION_HOME']}" \
                 f"/data/compositionality/nli/SETI/1.task1-primitive-composition/task1.2/composition/" \
                        f"train+val_PI_ver.json"

SETI_TASK_1_TEST = f"{os.environ['GENERALIZATION_HOME']}" \
                   f"/data/compositionality/nli/SETI/1.task1-primitive-composition/task1.2/composition/test.json"


SETI_TASK_2_TRAIN = f"{os.environ['GENERALIZATION_HOME']}" \
                        f"/data/compositionality/nli/SETI/2.task2-composed-composition/task2.1/composition/" \
                    f"train+val_CI.json"
SETI_TASK_2_TEST = f"{os.environ['GENERALIZATION_HOME']}" \
                   f"/data/compositionality/nli/SETI/2.task2-composed-composition/task2.1/composition/test.json"

MONLI_PMONLI_TEST = f"{os.environ['GENERALIZATION_HOME']}/data/compositionality/nli/MoNLI/pmonli.jsonl"

MONLI_NMONLI_TEST = f"{os.environ['GENERALIZATION_HOME']}/data/compositionality/nli/MoNLI/nmonli_test.jsonl"

COLS_TO_REMOVE = ['veridical_label', 'sick_label', 'sent1', 'sent2', 'label', 'str_label']

LABEL_2_ID = {'entailment': 0, 'neutral': 1}
ID_2_LABEL = {v: k for k, v in LABEL_2_ID.items()}


def tokenize_data(_file: str, tokenizer: transformers.AutoTokenizer, sample_percent=1., task="task1"):
    def tokenize_function(examples, task="task1"):
        if task in ["pmonli", "nmonli"]:
            return tokenizer(examples['sentence1'],examples['sentence2'])
        return tokenizer(examples['sent1'],examples['sent2'])

    def convert_labels2ids(example, task="task1"):
        if task in ["pmonli", "nmonli"]:
            example['str_label'] = example['gold_label']
        else:
            example["str_label"] = example['label']
        example['label'] = LABEL_2_ID[example['str_label']]
        if 'verdical_label' in example:
            example['veridical_label'] = example['verdical_label']
            del example['verdical_label']
        return example

    data = [json.loads(x) for x in open(_file)]
    random.shuffle(data)
    data = data[:int(len(data)*sample_percent)]
    data_hf = datasets.Dataset.from_list(data)
    data_hf = data_hf.map(convert_labels2ids, fn_kwargs={"task": task})
    tokenized_data_hf = data_hf.map(tokenize_function, batched=True, fn_kwargs={"task": task})\
        .filter(lambda sample: sample['label'] in ID_2_LABEL.keys())
    return tokenized_data_hf


def compute_metrics_entailment(eval_pred):
    """
    just get the accuracy for the entailment class now that the dataset only has neutrals and no contradiction
    :param eval_pred:
    :return:
    """
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, labels=[0], average="micro")


def test_compositionality_task_1_2(tokenizer, model, task="task1"):
    """
    for SETI task 1, we have a composition of two types of NLI datasets: lexical NLI from SICK and veridical inference
    the test data is a composition of these datasets. Our goal is to see how well the models perform individually
    and then how well they perform on the compositional test. We are testing with task 1.2 because that controls for
    lexical similarity
    :return:
    """
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    if task == "task1":
        d = {"nat": SETI_TASK_1_TRAIN_NAT, "ver": SETI_TASK_1_TRAIN_VER, "comp": SETI_TASK_1_TEST}
    elif task == "task2":
        d = {"train": SETI_TASK_2_TRAIN, "test": SETI_TASK_2_TEST}
    elif task == "pmonli":
        d = {"test": MONLI_PMONLI_TEST}
    elif task == "nmonli":
        d = {"test": MONLI_NMONLI_TEST}
    else:
        raise NotImplementedError(f"not implemented {task}")
    for k, _file in d.items():
        tokenized_data = tokenize_data(_file=_file, tokenizer=tokenizer, sample_percent=1.0, task=task)
        trainer = transformers.Trainer(model=model, data_collator=data_collator,
                                       compute_metrics=compute_metrics_entailment)
        results = trainer.evaluate(tokenized_data)
        print("="*30)
        print(k)
        print("-"*30)
        print(results)


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="bert-base-cased-snli")
    parser.add_argument("--task", default="task1")
    args = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"varun-v-rao/{args.model}")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(f"varun-v-rao/{args.model}")
    test_compositionality_task_1_2(tokenizer=tokenizer, model=model, task=args.task)


if __name__  == "__main__":
    main()



