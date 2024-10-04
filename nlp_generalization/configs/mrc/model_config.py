from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Attributes:
        model_checkpoint (str): Path to pretrained model or model identifier from huggingface.co/models.
        base_model (Optional[str]): The baseline model being used during training. Default is None.
        model_name (Optional[str]): The name of the model being trained (e.g., bert-base-cased-squad-model1). Default is None.
        model_class (Optional[str]): The architecture family that the model belongs to (e.g., bert, roberta, t5, opt). Default is None.
        tokenizer_name (Optional[str]): Pretrained tokenizer name or path if not the same as model_name. Default is None.
    """

    model_checkpoint: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_model: Optional[str] = field(
        default=None,
        metadata={"help": "The baseline model being used during training"}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the model being trained (e.g., bert-base-cased-squad-model1)"}
    )
    model_class: Optional[str] = field(
        default=None,
        metadata={"help": "The architecture family that the model belongs to (e.g., bert, roberta, t5, opt)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
          
    def __post_init__(self):
        if self.base_model is None:
            self.base_model = self.model_checkpoint.split('/')[-1]
        
        if self.model_name is None:
            self.model_name = self.model_checkpoint.split('/')[-1]
        
        if self.model_class is None:
            self.model_class = self.model_name.split('-')[0] 

        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_checkpoint