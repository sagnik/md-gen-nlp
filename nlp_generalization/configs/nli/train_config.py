from typing import Optional
from dataclasses import dataclass, field

@dataclass
class TrainConfig:    
    train_batch_size: Optional[int] = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    eval_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
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
    metric_name: Optional[str] = field(
        default="accuracy",
        metadata={"help": "Name of the metric used to determine best model checkpoint"}
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
        default="/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/nlp-gen/nli",
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