from datasets import load_dataset


def load_squad():
    return load_dataset("varun-v-rao/squad", split="validation")


def load_newsqa():
    return load_dataset("varun-v-rao/newsqa", split="validation")


def load_hotpotqa():
    return load_dataset("varun-v-rao/adversarial_hotpotqa", split="validation")


def load_musique():
    return load_dataset("sagnikrayc/musique-squad", split="test")


def load_musique():
    return load_dataset("sagnikrayc/musique-squad", split="test")


def load_musique():
    return load_dataset("sagnikrayc/musique-squad", split="test")


def load_adv_squad():
    return load_dataset("stanfordnlp/squad_adversarial", "AddSent", split="validation")


DATASET_LOADER_MAP = {
    "varun-v-rao/squad": load_squad,
    "varun-v-rao/newsqa": load_newsqa,
    "varun-v-rao/adversarial_hotpotqa": load_hotpotqa,
    "sagnikrayc/musique-squad": load_musique,
    "squad_adversarial": load_adv_squad,
}
