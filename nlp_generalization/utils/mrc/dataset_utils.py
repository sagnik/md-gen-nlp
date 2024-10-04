from datasets import load_dataset

def load_squad(split):
    return load_dataset("varun-v-rao/squad", split=split)

def load_newsqa(split):
    return load_dataset("varun-v-rao/newsqa", split=split)

def load_hotpotqa(split):
    return load_dataset("varun-v-rao/adversarial_hotpotqa", split=split)

def load_musique(split):
    return load_dataset("sagnikrayc/musique-squad", split=split)

DATASET_LOADER_MAP = {
    "squad": load_squad,
    "newsqa": load_newsqa,
    "adversarial_hotpotqa": load_hotpotqa,
    "musique": load_musique
}

def filter_by_length(examples, tokenizer, model: str):
    """
    Filter out samples that, when tokenized, are longer than the model's maximum input length.

    Args:
        examples (dict): Dictionary containing questions and contexts.
        tokenizer (Tokenizer): Tokenizer object for tokenizing the dataset.
        model (str): Name of the model being used (e.g., 'bert', 'roberta', 'opt').

    Returns:
        bool: True if the input length is less than the model's maximum input length, False otherwise.
    """
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
      
def tokenize_function(examples, tokenizer, model: str):
    """
    Tokenize function for preprocessing the dataset.

    Args:
        examples (dict): Non-preprocessed dataset containing questions and contexts.
        tokenizer (Tokenizer): Tokenizer object for tokenizing the dataset.
        model (str): Name of the model being used (e.g., 't5', 'bert', 'roberta').

    Returns:
        dict: Tokenized inputs with offset mappings and example IDs.
    """
    # Check if tokenizing for T5 or OPT models
    if model.startswith("t5") or model.startswith("opt"):
        questions = examples["question"]
        contexts = examples["context"]

        def generate_question_context_pairs(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        question_context_pairs = [generate_question_context_pairs(question, context) for question, context in zip(questions, contexts)]

        inputs = tokenizer(
            question_context_pairs,
            return_offsets_mapping=True,
        )

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 0 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = examples["id"]

        return inputs
    
    # Else tokenize for BERT/RoBerta models
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

def get_preprocessed_dataset(
    tokenizer, dataset_config, model_config, split: str = "train"
):
    """
    Build preprocessed dataset for training or evaluation.

    Args:
        tokenizer (Tokenizer): Tokenizer object for tokenizing the dataset.
        dataset_config (DatasetConfig): Configuration for the dataset.
        model_config (ModelConfig): Configuration for the model.
        split (str, optional): Split of the dataset to use ('train' or 'eval'). Defaults to 'train'.

    Returns: 
        datasets.Dataset: Processed dataset.
    """
    if not dataset_config.dataset_name in DATASET_LOADER_MAP:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")
        
    def get_split():
        return(
            dataset_config.train_split
            if split == "train"
            else dataset_config.eval_split
        )
    
    # Load raw dataset
    dataset = DATASET_LOADER_MAP[dataset_config.dataset_name](
        get_split()
    )
    
    # Filter by length
    dataset_filtered_by_length = dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "model": model_config.model_name}
    )
    
    # Tokenize
    tokenized_dataset = dataset_filtered_by_length.map(
        tokenize_function, 
        batched=True, 
        num_proc=dataset_config.preprocessing_num_workers, 
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "model": model_config.model_name}
    )
    
    return tokenized_dataset