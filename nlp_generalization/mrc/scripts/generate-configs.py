from jinja2 import Template
from huggingface_hub import HfApi
import re
from collections import defaultdict
from yaml import safe_load
import os
from rich.console import Console

API = HfApi()

def get_num_params(base_model_dict):
    base_model_dict_inv = {v:k for k,v in base_model_dict.items()}
    models = API.list_models(author="varun-v-rao")
    pattern = "-\d+\.*\d+[M|K]-"
    d = defaultdict(set)
    for model in models:
        if "squad" in model.id and re.findall(pattern, model.id):
            num_params = re.findall(pattern, model.id)[0][1:-1]
            base_model = None
            for model_str in base_model_dict_inv:
                if model_str.split('/')[-1] in model.id:
                    base_model = model_str
                    break
            assert base_model is not None, f"base model can not be none for {model.id}"
            d[num_params].add(base_model_dict_inv[base_model])
    return d


def render_and_write_file(base_dir, f_name, template, rendering_dict):
    def validate_rendered_template(_rendered: str):
        _d = safe_load(_rendered)
        if type(_d) != dict:
            return False
        model_name = _d["model"]["model_name"]
        # does the model exist?
        hf_models = [y for y in [x.id for x in API.list_models(search=model_name)] if y == model_name]
        if len(hf_models) == 1:
            return True
        elif len(hf_models) == 0:
            ## Debug ##
            print(model_name)
            ##################
            return False
        else:
            raise RuntimeError(f"multiple instances for {model_name}, {', '.join([x.id for x in hf_models])}")

    rendered = template.render(rendering_dict)
    if validate_rendered_template(rendered):
        with open(f"{base_dir}/{f_name}.yml", "w") as wf:
            wf.write(rendered)
    else:
        print(f"{f_name} could not be written")


def main():
    template = Template(open("configs/base-config.j2").read())
    base_dir = f"configs/generated-configs"
    os.makedirs(base_dir, exist_ok=True)
    model_types = {k:k for k in ["plain", "adapter", "lora"]}
    datasets = {'squad':'varun-v-rao/squad', 'newsqa':'varun-v-rao/newsqa', 'adversarial_hotpotqa':'varun-v-rao/adversarial_hotpotqa', 'musique': 'sagnikrayc/musique-squad'}
    base_models = {"bertbc": "bert-base-cased",
                   "bertlc": "bert-large-cased",
                   "rb": "roberta-base",
                   "rl": "roberta-large",
                   "t5b": "t5-base",
                   "t5l": "t5-large",
                   "optb": "facebook/opt-350m",
                   "optl": "facebook/opt-1.3b"
                   }
    num_params = get_num_params(base_models)
    model_indices = {str(k): str(k) for k in range(1, 4)}
    d = {
        "dataset": None,
        "base_model": None,
        "model_type": None,
        "num_params": None,
        "model_index": None
    }
    console = Console()
    with console.status("Working..."):
        for model_type_id, model_type_name in model_types.items():
            d["model_type"] = model_type_name
            for dataset_id, dataset_name in datasets.items():
                d["dataset"] = dataset_name
                for base_model_id, base_model_name in base_models.items():
                    d["tokenizer_name"] = base_model_name
                    d["base_model"] = base_model_name.split('/')[-1]
                    for model_index_id, model_index_name  in model_indices.items():
                        d["model_index"] = model_index_name
                        if model_type_name == "plain":
                            f_name = f"{dataset_id}-{base_model_id}-{model_type_id}-{model_index_id}"
                            render_and_write_file(base_dir, f_name, template, d)
                        else:
                            for num_param_id, num_param_models in num_params.items():
                                if base_model_id in num_param_models:
                                    d["num_params"] = num_param_id
                                    f_name = f"{dataset_id}-{base_model_id}-{model_type_id}-{num_param_id}-{model_index_id}"
                                    render_and_write_file(base_dir, f_name, template, d)


if __name__ == "__main__":
    main()