from adapters import AutoAdapterModel
from transformers import AutoModelForSequenceClassification


def load_adapter_model(base_model: str, model_name: str, **kwargs):
    """
    :param base_model:
    :param model_name:
    :return:
    """
    model = AutoAdapterModel.from_pretrained(base_model)
    model.load_adapter(model_name, source="hf", set_active=True)
    return model


def load_plain_model(model_name: str, **kwargs):
    """
    model_name
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name)


def load_lora_model(model_name: str, **kwargs):
    """
    model_name
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


MODEL_LOADER_MAP = {
    "adapter": load_adapter_model,
    "plain": load_plain_model,
    "lora": load_lora_model
}