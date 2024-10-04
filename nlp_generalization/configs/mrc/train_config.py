from typing import Optional
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    """
    Arguments pertaining to training.

    Attributes:
        train_batch_size (Optional[int]): Batch size per GPU/TPU/MPS/NPU core/CPU for training. Default is 16.
        eval_batch_size (Optional[int]): Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation. Default is 4.
        eval_accumulation_steps (Optional[int]): Number of prediction steps to accumulate before moving the tensors to the CPU. Default is 256.
        learning_rate (Optional[float]): The initial learning rate for the optimizer. Default is 2e-5.
        weight_decay (Optional[float]): Weight decay for the optimizer. Default is 0.01.
        num_train_epochs (Optional[int]): Total number of training epochs to perform. Default is 3.
        num_models_to_train (Optional[int]): Total number of model instances to train. Default is 3.
        seed (Optional[int]): Random seed that will be set at the beginning of training. Default is 42.
        use_random_seed (Optional[bool]): Whether or not to use a random seed. Default is True.
        shuffle_train_data (Optional[bool]): Whether or not to shuffle the training data. Default is True.
        checkpoints_save_path (Optional[str]): The output directory where model checkpoints will be written. Default is '/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/nlp-gen/qa'.
        checkpoint_dir (Optional[str]): Model output directory (helper attribute). Default is None.
        overwrite_output_dir (Optional[bool]): Whether or not to overwrite the output directory. Default is True.
        hf_org_name (Optional[str]): Hugging Face organization name. Default is 'varun-v-rao'.
        hf_write_token (Optional[str]): Hugging Face write token for pushing models to the hub. Default is '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'.
        hf_hub_id (Optional[str]): Hugging Face hub ID. Default is None.
        push_to_hub (Optional[bool]): Whether to push models to the Hugging Face Model Hub. Default is True.
        use_peft (Optional[bool]): Whether to use the LORA training strategy. Default is False.
        use_adapter (Optional[bool]): Whether to use the ADAPTER training strategy. Default is False.
        training_stratergy (Optional[str]): Training strategy used ('PLAIN', 'LORA', 'ADAPTER'). Default is 'PLAIN'.

    Raises:
        AssertionError: If both 'use_peft' and 'use_adapter' are set to True.

    Notes:
        - Only one of 'use_peft' or 'use_adapter' can be True at a given time.
        - Based on the selected strategy, 'training_strategy' is automatically set accordingly.
    """
    
    train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    eval_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    eval_accumulation_steps: Optional[int] = field(
        default=256,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."}
    )
    learning_rate: Optional[float] = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for the optimizer."}
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "Weight decay for the optimizer."}
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    num_models_to_train: Optional[int] = field(
        default=3,
        metadata={"help": "Total number of model instances to train."}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    use_random_seed: Optional[bool] = field(
        default=True,
        metadata={"help":"Whether or not to use random seed."}
    )
    shuffle_train_data: Optional[bool] = field(
        default=True,
        metadata={"help":"Whether or not to shuffle the training data"}
    )
    checkpoints_save_path: Optional[str] = field(
        default="/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/nlp-gen/qa",
        metadata={"help": "The output directory where model checkpooints will be written"}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Model output dir (helper attribute)"}
    )
    overwrite_output_dir : Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to overwrite the output dir"}
    )
    hf_org_name: Optional[str] = field(
        default="varun-v-rao",
        metadata={"help":"Hugging Face organization name."}
    )
    hf_write_token: Optional[str] = field(
        default="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",
        metadata={"help":"Hugging Face write token for pushing models to the hub."}
    )
    hf_hub_id: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face hub ID."}
    )
    push_to_hub: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to push models to the Hugging Face Model Hub."}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LORA training stratergy."}
    )
    use_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use ADAPTER training stratergy"}
    )
    training_stratergy: Optional[str] = field(
        default="PLAIN",
        metadata={"help": "Training strategy used ('PLAIN', 'LORA', 'ADAPTER')."}
    )
        
    def __post_init__(self):
        # Assert only one of peft or adapter is True at a given time
        assert not (self.use_peft and self.use_adapter), "Please ensure that only one of the 'peft' or 'adapter' flags is selected at any given time."
        if self.use_peft:
            self.training_stratergy = "LORA"
        elif self.use_adapter:
            self.training_stratergy = "ADAPTER"
        else:
            self.training_stratergy = "PLAIN"