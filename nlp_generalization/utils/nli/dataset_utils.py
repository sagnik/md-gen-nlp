"""
To-Do:

Update Dataset_Config to include validation_split and test_split. Also remove eval_split
"""

from datasets import load_dataset

ID_2_LABEL = {0:'entailment', 1:'neutral', 2:'contradiction'}
LABEL_2_ID = {v: k for k, v in ID_2_LABEL.items()}
NUM_LABELS = len(ID_2_LABEL)

def convertlabels2ids(example):
    """
    Convert label names to corresponding label IDs.

    Args:
        example (Dict): Dictionary containing examples with 'label' key.

    Returns:
        Dict: Dictionary with label names converted to their corresponding label IDs.
    """
    example['label'] = LABEL_2_ID[example['label']]
    return example

def load_snli(split):
    return load_dataset("stanfordnlp/snli", split=split)
     
def load_mnli(split):
    return load_dataset("nyu-mll/multi_nli", split=split)

def load_snli_cf(split):
    data = load_dataset("sagnikrayc/snli-cf-kaushik", split=split)
    return data.map(convertlabels2ids)

def load_snli_bt(split):
    data = load_dataset("sagnikrayc/snli-bt", split=split)
    return data.map(convertlabels2ids)

def load_sick(split):
    dataset = load_dataset("RobZamp/sick", split=split)
    return dataset.rename_column("sentence_A", "premise").rename_column("sentence_B", "hypothesis")

DATASET_LOADER_MAP = {
    "snli": load_snli,
    "multi_nli": load_mnli,
    "snli-cf-kaushik": load_snli_cf,
    "snli-bt": load_snli_bt,
    "sick": load_sick
}

def filter_by_length(examples, tokenizer, model_name:str):
    """
    Filter out samples based on their length after tokenization.

    Args:
        examples (Dict): Dictionary containing examples with 'premise' and 'hypothesis' keys.
        tokenizer (Tokenizer): Tokenizer object for tokenizing the examples.
        model_name (str): Name of the model being used.

    Returns:
        bool: True if the combined length of premise and hypothesis after tokenization is less than the model's maximum input length, otherwise False.
    """
    premise = examples['premise']
    hypothesis = examples['hypothesis']
    
    if model_name.startswith("bert") or model_name.startswith("roberta"):
        input_len = len(tokenizer(premise,hypothesis, truncation=False).input_ids)
        return input_len < tokenizer.model_max_length
    else:
        def generate_input(_premise, _hypothesis):
            return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip()])
        
        input_len = len(tokenizer(generate_input(premise, hypothesis), truncation=False).input_ids)
        return (input_len < 512) # OPT models dont have model_max_length defined, so I fix it to 512
    
def filter_by_label(examples):
    """
    Filter out samples based on their labels.

    Args:
        examples (Dict): Dictionary containing examples with 'label' key.

    Returns:
        bool: True if the label of the example is within the range of labels defined in ID_2_LABEL, otherwise False.
    """
    return examples['label'] in list(range(len(ID_2_LABEL)))

def preprocess_nli_batch(examples):
    """
    Preprocess natural language inference (NLI) examples.

    Args:
        examples (Dict): Dictionary containing examples with 'premise' and 'hypothesis' keys.

    Returns:
        List[str]: List of preprocessed input sequences.
    """
    premises = examples['premise']
    hypotheses = examples['hypothesis']

    def generate_input(_premise, _hypothesis):
        return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip()])

    inputs = [generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
    return inputs

def tokenize_function(examples, tokenizer, model_name:str):
    """
    Tokenize examples based on the model type.

    Args:
        examples (Dict): Dictionary containing examples with 'premise' and 'hypothesis' keys.
        tokenizer (Tokenizer): Tokenizer object for tokenizing the examples.
        model_name (str): Name of the model being used.

    Returns:
        Union[List[str], Tuple[List[str], List[str]]]: List of tokenized inputs.
    """
    if model_name.startswith("t5") or model_name.startswith("opt"):
        return tokenizer(preprocess_nli_batch(examples))
    return tokenizer(examples['premise'], examples['hypothesis'])


def get_preprocessed_dataset(
    tokenizer, dataset_config, model_config, split: str = "train"
):
    """
    Build preprocessed dataset for training or evaluation.

    Args:
        tokenizer (Tokenizer): Tokenizer object used for tokenization.
        dataset_config (DatasetConfig): Dataset configuration object.
        model_config (ModelConfig): Model configuration object.
        split (str): Split type of the dataset, either 'train' or 'eval'.

    Returns:
        datasets.Dataset: Preprocessed dataset ready for training or evaluation.
    """
    if not dataset_config.dataset_name in DATASET_LOADER_MAP:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")
       
    # Unused, delete later?
    def get_split():
        if split == "train":
            return dataset_config.train_split
        elif split == "validation":
            return dataset_config.validation_split
        else:
            return dataset_config.test_split
    
    # Load raw dataset
    dataset = DATASET_LOADER_MAP[dataset_config.dataset_name](
        split#get_split()
    )
    
    # Filter by length
    dataset_filtered_by_length = dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_config.model_name}
    )
    
    # Filter by label
    dataset_filtered = dataset_filtered_by_length.filter(filter_by_label)
    
    # Tokenize
    tokenized_dataset = dataset_filtered.map(
        tokenize_function,
        batched=True,
        num_proc=dataset_config.preprocessing_num_workers,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_config.model_name}
    )
    
    # Remove unused columns
    col_names = dataset.column_names
    col_names.remove('label')
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels').remove_columns(col_names)
    
    return tokenized_dataset
        
