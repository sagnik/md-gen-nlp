"""
Script to test musique performance on trained models.
We will choose the questions that are composition of Squad questions. We will output the performance for the component
squad questions and finally the performance on the composed questions.
"""
import os

from pydantic import BaseModel
from typing import Dict, List
from nlp_generalization.mrc.scripts.run_qa import compute_metrics
from datasets import Dataset, load_dataset
import argparse
import transformers
from transformers import DefaultDataCollator
import json
import nlp_generalization.yamlenv as yamlenv

class BaseQuestionContext(BaseModel):
    wikipedia_id: int
    wikipedia_title: str
    paragraph_indices: List[int]
    paragraph_text: str
    is_supporting: bool


class BaseQuestion(BaseModel):
    id: int
    dataset: str
    answerable: bool
    question_text: str
    answer_text: str
    is_tail: bool
    question_reverse_replacement: Dict[str, str]
    contexts: List[BaseQuestionContext]
    query_text: str


class ComposedQuestionContext(BaseQuestionContext):
    retrieval_score: float
    primary: bool


class ComposedQuestion(BaseModel):
    dataset: str
    setname: str
    id: str
    decomposed_instances: List[BaseQuestion]
    question_text: str
    component_question_texts: List[str]
    query_text: str
    component_query_texts: List[str]
    answer_text: str
    answerable: bool
    contexts: List[ComposedQuestionContext]
    composed_question_text: str


class CompositionalResult(BaseModel):
    id: str
    composite_em: float
    composite_f1: float
    subq_em: float
    subq_f1: float


def read_composed_question(cq: ComposedQuestion) -> List[Dict]:
    """
    We will return a list of examples that can be tokenized -- they will look like
    extractive squad examples. The first example will be the composed question. The rest will
    be the questions that compose that first question.
    :param cq:
    :return:
    """
    def replace_q(_item: BaseQuestion):
        org_q = _item.question_text
        for k, v in _item.question_reverse_replacement.items():
            org_q = org_q.replace(k, v)
        return org_q

    def get_squad_format(item: ComposedQuestion | BaseQuestion) -> Dict:
        context = "\n".join([x.paragraph_text for x in item.contexts if x.is_supporting])
        answer_text = item.answer_text
        answer_start = context.index(answer_text)  # deliberately throw ValueError if not found.
        if type(item) == BaseQuestion:
            question = replace_q(item)
        else:
            question = item.composed_question_text
        return {
            "question": question,
            "title": "NA",
            "id": str(item.id),
            "context": context,
            "answers": {"text": [answer_text], "answer_start": [answer_start]}
        }

    # is this question answerable? are all sub questions answerable?
    if not cq.answerable or any([not x.answerable for x in cq.decomposed_instances]):
        return []

    cq_squad_fmt = get_squad_format(cq)
    subq_squad_fmt = [get_squad_format(x) for x in cq.decomposed_instances]
    return [cq_squad_fmt] + subq_squad_fmt


def filter_function(examples, tokenizer):
    question = examples["question"]
    context = examples["context"]
    input_len = len(tokenizer(question,context, truncation=False).input_ids)
    return input_len < tokenizer.model_max_length


def tokenize(examples, tokenizer):
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


def calc_result(_tokenized_data, _test_data, trainer):
    predictions, _, _ = trainer.predict(_tokenized_data)
    start_logits, end_logits = predictions
    results = compute_metrics(start_logits, end_logits, _tokenized_data, _test_data)
    return results


def evaluate_adv_squad(tokenizer, trainer, exp_params):
    test_data = load_dataset(exp_params["path"], exp_params["setup"], split=exp_params["split"], trust_remote_code=True)
    test_data_filtered = test_data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    print(f"original: {len(test_data)}, filtered: {len(test_data_filtered)}, "
          f"{len(test_data) - len(test_data_filtered)} data points filtered out due to length")
    tokenized_test_data = test_data_filtered.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    result = calc_result(_tokenized_data=tokenized_test_data, _test_data=test_data_filtered, trainer=trainer)

    output_dir, _ = os.path.split(exp_params["output"])
    os.makedirs(output_dir, exist_ok=True)
    json.dump(result, open(exp_params["output"], "w"), indent=2)


def evaluate_musique(tokenizer, trainer, exp_params):
    """
    Generate
    :return:
    """
    data = [ComposedQuestion(**json.loads(x)) for x in open(exp_params['dataset_loc'])]
    print(f"loaded {len(data)} data points")
    all_results = []
    num_long = 0
    num_samples = exp_params.get('num_samples', len(data))
    for item in data[:num_samples]:
        sqd_fmt_data = read_composed_question(item)
        if not sqd_fmt_data:  # empty question
            continue
        test_data = Dataset.from_list(sqd_fmt_data)
        if len(test_data) != len(test_data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})):
            # if any one of this set gets rejected for length, that's definitely the compositional one.
            print("no compositional question to evaluate on")
            num_long += 1
            continue
        test_comp, test_subq = Dataset.from_list(sqd_fmt_data[:1]), Dataset.from_list(sqd_fmt_data[1:])
        tokenized_test_comp = test_comp.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
        tokenized_test_subq = test_subq.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
        comp_result = calc_result(_tokenized_data=tokenized_test_comp, _test_data=test_comp, trainer=trainer)
        subq_result = calc_result(_tokenized_data=tokenized_test_subq, _test_data=test_subq, trainer=trainer)
        all_results.append(
            CompositionalResult(
                id=item.id,
                composite_em=comp_result['exact_match'],
                composite_f1=comp_result['f1'],
                subq_em=subq_result['exact_match'],
                subq_f1=subq_result['f1']
            )
        )
    print(f"length error: {num_long} out of {len(data)}")

    output_dir, _ = os.path.split(exp_params["output"])
    os.makedirs(output_dir, exist_ok=True)

    with open(exp_params["output"], "w") as wf:
        for item in all_results:
            wf.write(item.model_dump_json()+"\n")


DATASET_EVAL_FN_MAP = {"musique": evaluate_musique, "adv_squad": evaluate_adv_squad}


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/config-musique.yml")
    args = parser.parse_args()

    config = yamlenv.load(open(args.config))
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(config["tokenizer"])
    data_collator = DefaultDataCollator()
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    evaluate_fn = DATASET_EVAL_FN_MAP[config['dataset']]
    evaluate_fn(tokenizer=tokenizer, trainer=trainer, exp_params=config['exp_params'])


if __name__ == "__main__":
    main()
