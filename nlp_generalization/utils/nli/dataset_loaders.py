from datasets import load_dataset, Dataset
import pandas as pd
from collections import defaultdict as ddict
import json
import os
from tqdm import tqdm

def load_snli(sample=False):
    """
    uncomment for sampling
    :return:
    """
    data = load_dataset("snli", split="test", trust_remote_code=True)
    if not sample:
        return data
    return data.train_test_split(test_size=64)["test"]


def load_mnli():
    return load_dataset("multi_nli", split="validation_mismatched", trust_remote_code=True)

def load_matmnli():
    return load_dataset("multi_nli", split="validation_matched", trust_remote_code=True)


def load_data_from_json(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            d = json.loads(line)
            dataset.append(d)
    return dataset


def convert_labels_2_ids(example):
    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    label2id = {v: k for k, v in id2label.items()}
    example['label'] = label2id[example['label']]
    return example


def load_hans():
    return load_dataset("hans", split="validation", trust_remote_code=True)


def load_snli_cf():
    data = load_dataset("sagnikrayc/snli-cf-kaushik", split="test", trust_remote_code=True)
    return data.map(convert_labels_2_ids)


def load_snli_bt():
    data = load_dataset("sagnikrayc/snli-bt", split="test", trust_remote_code=True)
    return data.map(convert_labels_2_ids)


def load_snli_hard():
    dataset = load_dataset("au123/snli-hard", split="test", trust_remote_code=True)
    
    snli_hard_dict = ddict(list)
    premise_sents       = []
    hypothesis_sents    = []
    idx                 = 0
    
    for row in dataset:
        snli_hard_dict['promptID'].append(f'SNLI_hard_{idx}')
        snli_hard_dict['pairID'].append(row['pairID'])
        snli_hard_dict['label'].append(row['gold_label'])
        snli_hard_dict['premise'].append(row['sentence1'])
        snli_hard_dict['hypothesis'].append(row['sentence2'])
        idx += 1
        
    ds = Dataset.from_dict(snli_hard_dict)
    return ds.map(convert_labels_2_ids)


def load_conjnli():
    dataset = pd.read_csv('/data/shire/projects/generalization/nlp_generalization/utils/nli/downloaded_datasets/conj_dev.tsv', sep='\t')

    st_dict = ddict(list)
    premise_sents       = []
    hypothesis_sents    = []

    for idx, row in dataset.iterrows():

        st_dict['promptID'].append(f'ConjNLI_{idx}')
        st_dict['pairID'].append(f'ConjNLI_{idx}_{row["Label"][0]}')
        st_dict['label'].append(row['Label'])
        st_dict['premise'].append(row['Premise'])
        st_dict['hypothesis'].append(row['Hypothesis'])
        
        st_dict['file_name'].append('ConjNLI')
    
    ds = Dataset.from_dict(st_dict)
    return ds.map(convert_labels_2_ids)


def load_stress_test():
    st_dir          = '/data/shire/projects/generalization/nlp_generalization/utils/nli/downloaded_datasets/Stress_Tests'
    st_dict         = ddict(list)
    file_idx        = 0

    for dir2 in os.listdir(st_dir):
        st_type     = dir2
        if not os.path.isdir(f'{st_dir}/{dir2}'):continue

        for file in os.listdir(f'{st_dir}/{dir2}'):

            if file.endswith('.jsonl'):
                data = open(f'{st_dir}/{dir2}/{file}').readlines()
                
                idx                 = 0
                for line in tqdm(data):
                    idx             +=1
                    elem   = json.loads(line)

                    if st_type == 'Numerical_Reasoning':
                        st_dict['promptID'].append(f'{st_type}_{idx}')
                        st_dict['pairID'].append(f'{st_type}_{idx}_{elem["gold_label"][0]}')
                        st_dict['label'].append(elem['gold_label'])
                    else:
                        st_dict['promptID'].append(elem['promptID'])
                        st_dict['pairID'].append(elem['pairID'])
                        st_dict['label'].append(elem['gold_label'])
                    
                    st_dict['premise'].append(elem['sentence1'])
                    st_dict['hypothesis'].append(elem['sentence2'])
                    
                    st_dict['stress_test_type'].append(st_type)
                    st_dict['file_name'].append(file)
    
    ds = Dataset.from_dict(st_dict)
    return ds.map(convert_labels_2_ids)



def load_taxinli():
    taxi_data           = pd.read_csv('/data/shire/projects/generalization/nlp_generalization/utils/nli/downloaded_datasets/taxinli_dev.tsv', sep='\t')
    st_dict             = ddict(list)
    premise_sents       = []
    hypothesis_sents    = []

    features            = ['lexical_linguistic', 'syntactic_linguistic', 	'factivity_linguistic', 'negation_logic', 	'boolean_logic',\
                    'quantifier_logic',	'conditional_logic',	'comparative_logic',	'relational_reasoning',	'spatial_reasoning',\
                    'temporal_reasoning',	'causal_reasoning',	'coreference_reasoning',	'world_knowledge',	'taxonomic_knowledge']

    for idx, row in taxi_data.iterrows():
        for feature in features:
            if row[feature] == 1:
                st_dict['promptID'].append(row['pairID'][:-1])
                st_dict['pairID'].append(row['pairID'])
                st_dict['label'].append(row['label'])
                st_dict['premise'].append(row['prem'])
                st_dict['hypothesis'].append(row['hyp'])
                st_dict['feature'].append(feature)
    
    ds = Dataset.from_dict(st_dict)
    return ds.map(convert_labels_2_ids)               


def load_pmonli():
    
    pmonli_jdict = load_data_from_json('/data/shire/projects/generalization/nlp_generalization/utils/nli/downloaded_datasets/pmonli.jsonl')
    
    st_dict         = ddict(list)
    
    for idx, elem in enumerate(pmonli_jdict):
        st_dict['pairID'].append(f'pmonli_{idx}_{elem["gold_label"][0]}')
        st_dict['premise'].append(elem['sentence1'])
        st_dict['hypothesis'].append(elem['sentence2'])
        st_dict['label'].append(elem['gold_label'])
    
    ds = Dataset.from_dict(st_dict)
    return ds.map(convert_labels_2_ids)


def load_nmonli():
    
    nmonli_jdict = load_data_from_json('/data/shire/projects/generalization/nlp_generalization/utils/nli/downloaded_datasets/nmonli_test.jsonl')
    
    st_dict         = ddict(list)
    
    for idx, elem in enumerate(nmonli_jdict):
        st_dict['pairID'].append(f'nmonli_{idx}_{elem["gold_label"][0]}')
        st_dict['premise'].append(elem['sentence1'])
        st_dict['hypothesis'].append(elem['sentence2'])
        st_dict['label'].append(elem['gold_label'])
    
    ds = Dataset.from_dict(st_dict)
    return ds.map(convert_labels_2_ids)






DATASET_LOADER_MAP = {
    "snli": load_snli,
    "mnli": load_mnli,
    "matmnli": load_matmnli,
    "snli-cf-kaushik": load_snli_cf,
    "snli-bt": load_snli_bt,
    "hans": load_hans,
    "taxinli": load_taxinli,
    "conjnli": load_conjnli,
    "stress-test": load_stress_test,
    "pmonli": load_pmonli,
    "nmonli": load_nmonli,
    'snli_hard': load_snli_hard
}
