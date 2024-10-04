from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DatasetArguments:    
    dataset_name: Optional[str] = field(
        default="snli", 
        metadata={"help": "The name of the dataset to use (e.g., 'snli', 'multi_nli', 'snli-cf-kaushik', 'snli-bt')."}
    )
    train_split: Optional[str] = field(
        default="train", 
        metadata={"help": "The dataset split to use for training the model"}
    )
    eval_split: Optional[str] = field(
        default="validation", 
        metadata={"help": "The dataset split to use for validating model performance"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."}
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."}
    )