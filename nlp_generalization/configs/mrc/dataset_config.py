from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training or eval.

    Attributes:
        dataset_name (Optional[str]): The name of the dataset to use (e.g., 'squad', 'newsqa', 'adversarial_hotpotqa'). Default is 'squad'.
        train_split (Optional[str]): The dataset split to use for training the model. Default is 'train'.
        eval_split (Optional[str]): The dataset split to use for validating model performance. Default is 'validation'.
        preprocessing_num_workers (Optional[int]): The number of processes to use for the preprocessing. Default is None.
        max_seq_length (Optional[int]): The maximum total input sequence length after tokenization. Default is 512.
        max_train_samples (Optional[int]): For debugging purposes or quicker training, truncate the number of training examples to this value if set. Default is None.
        max_eval_samples (Optional[int]): For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set. Default is None.
    """
    
    dataset_name: Optional[str] = field(
        default="squad", 
        metadata={"help": "The name of the dataset to use (e.g., 'squad', 'newsqa', 'adversarial_hotpotqa')."}
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."}
    )