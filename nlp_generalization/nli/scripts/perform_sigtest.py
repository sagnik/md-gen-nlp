"""
nli models
"""
import os.path
import re
import evaluate
import numpy as np
from collections import defaultdict as ddict
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import  accuracy_score, f1_score
import random
import math
from pprint import pprint
import pickle
from scipy.stats import spearmanr, kendalltau, pearsonr, kstest
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import json
from mc4.algorithm import mc4_aggregator
import matplotlib.pyplot as plt 


def compute_bootstrapped_df(df1, df2, df1_name, df2_name, num_iter = 1000):
    
    assert len(df1) == len(df2)
    
    idx_list = [idx for idx in range(len(df1))]
    
    cnt = 0
    
    gold_lbl1s = df1['label'].tolist()
    gold_lbl2s = df2['label'].tolist()
    
    df1_lbls  = df1[df1_name].tolist()
    df2_lbls  = df2[df2_name].tolist()
    
    df1_acc = accuracy_score(gold_lbl1s, df1_lbls)
    df2_acc = accuracy_score(gold_lbl2s, df2_lbls)
        
    sys_diff = df2_acc - df1_acc
    
    if sys_diff <= 0:
        temp_df, temp_name   = df2.copy(), df2_name
        df2, df2_name        = df1.copy(), df1_name
        df1, df1_name        = temp_df.copy(), temp_name
        # sys_diff             = -sys_diff
        
        gold_lbls = df1['label'].tolist()
        df1_lbls  = df1[df1_name].tolist()
        df2_lbls  = df2[df2_name].tolist()
        
        df1_acc = accuracy_score(gold_lbls, df1_lbls)
        df2_acc = accuracy_score(gold_lbls, df2_lbls)
            
        sys_diff = df2_acc - df1_acc
        
    assert sys_diff >= 0
    
    for i in range(0, num_iter):
        random.seed(i)
        np.random.seed(i)
        curr_idx_list = random.choices(idx_list, k=min(int(0.1*len(idx_list)), 1000))
        
        df1_pred = df1.iloc[curr_idx_list][df1_name].tolist()
        df2_pred = df2.iloc[curr_idx_list][df2_name].tolist()
        gold_lbl = df1.iloc[curr_idx_list]['label'].tolist()
        
        curr_df1_acc = accuracy_score(gold_lbl, df1_pred)
        curr_df2_acc = accuracy_score(gold_lbl, df2_pred)
        
        # comes from the 2012 non-paramteric method of boostrap significance testing
        curr_diff = curr_df2_acc - curr_df1_acc
        if curr_diff > 2*sys_diff:
            cnt += 1
    
    sig_val = ''
    p_val = cnt/num_iter

    if p_val < 0.001 :
        sig_val = '***'
    elif p_val < 0.01: 
        sig_val = '**'
    elif p_val < 0.05:
        sig_val = '*'
    
    if p_val > 0.999:
        sig_val = '---'
    elif p_val > 0.99:
        sig_val = '--'
    elif p_val > 0.95:
        sig_val = '-'
    
    results_dict = {
        'p_val': p_val, 
        'sig_val': sig_val,
        'better_model': df2_name,
        'better_model_acc': round(100*df2_acc,2),
        'worse_model': df1_name,
        'worse_model_acc': round(100*df1_acc,2)
    }
    
    return results_dict


def get_dataset_name_and_labels(file):
    if file.startswith('snli-cf'):
        dataset= 'snli-cf'
    elif file.startswith('snli-bt'):
        dataset= 'snli-bt'
    else:
        dataset = file.split('-')[0]
    
    df = pd.read_csv(os.path.join(args.inp_dir, file))
    
    labels = set(df['label'])
        
    return dataset, labels
    

def compute_datasetwise_results():
    
    with open('../out_files/nli_instancewise_results.json', 'r') as f:
        acc_dict = json.load(f)
    
    nli_datasets = acc_dict.keys()
    datasetwise_dict = ddict(list)
    
    for dataset in nli_datasets:
        if dataset == 'nmonli': continue
        accs        = []
        for model in acc_dict[dataset]:
            accs.append(acc_dict[dataset][model])
        
        datasetwise_dict['Dataset'].append(dataset)
        datasetwise_dict['Mean'].append(round(100*np.mean(accs),2))
        datasetwise_dict['Std'].append(round(100*np.std(accs),2))
    
    datasetwise_df = pd.DataFrame(datasetwise_dict)
    datasetwise_df.to_csv(f'../out_files/{args.task}_datasetwise_results.csv', index=False)


def create_nli_dict():
    acc_dict            = ddict(dict)
    datasetwise_dict    = ddict(list)
    
    nli_dict = ddict(lambda: ddict(list))
    dataset = None
    for file in tqdm(os.listdir(args.inp_dir), desc='Reading files'):
        assert file.endswith('.csv')
                
        dataset, labels = get_dataset_name_and_labels(file)
        seed    = file.split('-')[-1].split('.')[0]
        model   = file.replace(dataset, '').replace(f'{seed}.yml.csv', '')[1:-1]
        
        df                  = pd.read_csv(os.path.join(args.inp_dir, file))
        df['seed']          = [seed]*len(df)
        df['predictions']   = [row['predictions'] if row['predictions'] in labels else 1 for index, row in df.iterrows()]
        
        predictions         = list([row['predictions'] if row['predictions'] in labels else 1 for index, row in df.iterrows()])
        labels              = list([row['label'] for index, row in df.iterrows()])
        acc                 = accuracy_score(labels, predictions)
        
        acc_dict[dataset][f'{model}-{seed}'] = acc
        datasetwise_dict[dataset].append(acc)    
        nli_dict[dataset][model].append(df)
        
    
    with open(f'../out_files/{args.task}_instancewise_results.json', 'w') as wf:
        json.dump(acc_dict, wf)
    
    updated_nli_dict = ddict(dict)
    
    for dataset in nli_dict:
        for model in nli_dict[dataset]:
            comb_df = pd.concat(nli_dict[dataset][model])
            comb_df[model] = comb_df['predictions']
            updated_nli_dict[dataset][model] = comb_df
            
            gold_lbls, pred_lbls             = list(comb_df['label']), list(comb_df['predictions'])
            
    return updated_nli_dict
    

def perform_sig_test(nli_dict):
    
    analysis_dict = ddict(list)
    
    for dataset in nli_dict:
        for model1 in nli_dict[dataset]:
            for model2 in nli_dict[dataset]:
                if model1 != model2:
                    
                    if 'optb-adapter' not in model1 and 'optl-adapter' not in model1 and 'optb-adapter' not in model2 and 'optl-adapter' not in model2: continue
                    
                    results_dict = compute_bootstrapped_df(nli_dict[dataset][model1], nli_dict[dataset][model2], model1, model2, num_iter = 1000)
                    
                    analysis_dict['dataset'].append(dataset)
                    # analysis_dict['model1'].append(model1)
                    # analysis_dict['model2'].append(model2)
                    analysis_dict['p_val'].append(results_dict['p_val'])
                    analysis_dict['sig_val'].append(results_dict['sig_val'])
                    analysis_dict['better_model'].append(results_dict['better_model'])
                    analysis_dict['better_acc'].append(results_dict['better_model_acc'])
                    analysis_dict['worse_model'].append(results_dict['worse_model'])
                    analysis_dict['worse_acc'].append(results_dict['worse_model_acc'])
                    
                    
                    print(f"{dataset} {model1} vs {model2}: Better model is {results_dict['better_model']} with p-value: {results_dict['p_val']} {results_dict['sig_val']}")
                    
    
    analysis_df = pd.DataFrame(analysis_dict)
    analysis_df.to_csv(f'{args.task}_opt_adapter_sig_test.csv', index=False)

def validate_sig_hyps():
    
    
    hypothesis_cnt_dict = ddict(lambda: ddict(int))
    
    sig_df      = pd.read_csv(f'../out_files/{args.task}_sig_test.csv')
    
    models      = ['bertbc-adapter-895K', 'bertbc-lora-592K', 'bertbc-plain', 'bertlc-adapter-3.17M', 'bertlc-lora-1.58M', 'bertlc-plain', 'optb-adapter', 'optb-lora-1.57M', 'optb-plain', 'optl-adapter' 'optl-lora-3.15M', 'optl-plain',\
        'rb-adapter-895K', 'rb-lora-1.18M', 'rb-plain', 'rl-adapter-3.17M', 'rl-lora-2.63M', 'rl-plain', 't5b-adapter-1.79M', 't5b-lora-1.77M', 't5b-plain', 't5l-adapter-6.34M', 't5l-lora-4.72M', 't5l-plain']
    
    base_models = ['bertbc-adapter-895K', 'bertbc-lora-592K', 'bertbc-plain', 'optb-adapter', 'optb-lora-1.57M', 'optb-plain',  'rb-adapter-895K', 'rb-lora-1.18M', 'rb-plain',  't5b-adapter-1.79M', 't5b-lora-1.77M', 't5b-plain']
    
    large_models = ['bertlc-adapter-3.17M', 'bertlc-lora-1.58M', 'bertlc-plain', 'optl-adapter', 'optl-lora-3.15M', 'optl-plain', 'rl-adapter-3.17M', 'rl-lora-2.63M', 'rl-plain', 't5l-adapter-6.34M', 't5l-lora-4.72M', 't5l-plain']
    
    plain_models = ['bertbc-plain', 'bertlc-plain', 'optb-plain', 'optl-plain', 'rb-plain', 'rl-plain', 't5b-plain', 't5l-plain']
    
    lora_models  = ['bertbc-lora-592K', 'bertlc-lora-1.58M', 'optb-lora-1.57M', 'optl-lora-3.15M', 'rb-lora-1.18M', 'rl-lora-2.63M', 't5b-lora-1.77M', 't5l-lora-4.72M']
    
    adapter_models = ['bertbc-adapter-895K', 'bertlc-adapter-3.17M', 'optb-adapter', 'optl-adapter', 'rb-adapter-895K', 'rl-adapter-3.17M', 't5b-adapter-1.79M', 't5l-adapter-6.34M']
    
    eo_models = ['bertbc-adapter-895K', 'bertbc-lora-592K', 'bertbc-plain', 'bertlc-adapter-3.17M', 'bertlc-lora-1.58M', 'bertlc-plain', 'rb-adapter-895K', 'rb-lora-1.18M', 'rb-plain', 'rl-adapter-3.17M', 'rl-lora-2.63M', 'rl-plain']
    
    do_models = ['optb-adapter', 'optb-lora-1.57M', 'optb-plain', 'optl-adapter', 'optl-lora-3.15M', 'optl-plain']
    
    ed_models = ['t5b-adapter-1.79M', 't5b-lora-1.77M', 't5b-plain', 't5l-adapter-6.34M', 't5l-lora-4.72M', 't5l-plain']
    
    
    analysis_cnt_dict = ddict(lambda: ddict(int))
    
    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
        if dataset =='nmonli': continue
        
        for bm, lm in zip(base_models, large_models):
            if lm == better_model and bm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['large']['better'] += 1
                else:
                    hypothesis_cnt_dict['large']['equal'] += 1
            
            elif lm == worse_model and bm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['large']['worse'] += 1
                else:
                    hypothesis_cnt_dict['large']['equal'] += 1
    
        for pm, lm in zip(plain_models, lora_models):
            if lm == better_model and pm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['lora']['better'] += 1
                else:
                    hypothesis_cnt_dict['lora']['equal'] += 1
            
            elif lm == worse_model and pm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['lora']['worse'] += 1
                else:
                    hypothesis_cnt_dict['lora']['equal'] += 1
        
        for spm, am in zip(plain_models, adapter_models):
            if am == better_model and spm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter']['better'] += 1
                else:
                    hypothesis_cnt_dict['adapter']['equal'] += 1
            
            elif am == worse_model and spm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter']['worse'] += 1
                else:
                    hypothesis_cnt_dict['adapter']['equal'] += 1

        for am, lm in zip(adapter_models, lora_models):
            if am == better_model and lm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter vs lora']['better'] += 1
                else:
                    hypothesis_cnt_dict['adapter vs lora']['equal'] += 1
            
            elif am == worse_model and lm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter vs lora']['worse'] += 1
                else:
                    hypothesis_cnt_dict['adapter vs lora']['equal'] += 1
    
        

        if better_model in eo_models and worse_model in do_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['eo vs do']['better'] += 1
            else:
                hypothesis_cnt_dict['eo vs do']['equal'] += 1

        elif better_model in do_models and worse_model in eo_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['eo vs do']['worse'] += 1
            else:
                hypothesis_cnt_dict['eo vs do']['equal'] += 1
        
        if better_model in eo_models and worse_model in ed_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['eo vs ed']['better'] += 1
            else:
                hypothesis_cnt_dict['eo vs ed']['equal'] += 1
        
        elif better_model in ed_models and worse_model in eo_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['eo vs ed']['worse'] += 1
            else:
                hypothesis_cnt_dict['eo vs ed']['equal'] += 1

        if better_model in ed_models and worse_model in do_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['ed vs do']['better'] += 1
            else:
                hypothesis_cnt_dict['ed vs do']['equal'] += 1
        
        elif better_model in do_models and worse_model in ed_models:
            if p_val < 0.05:
                hypothesis_cnt_dict['ed vs do']['worse'] += 1
            else:
                hypothesis_cnt_dict['ed vs do']['equal'] += 1
    
    
    pprint(hypothesis_cnt_dict)
        

def get_input_categories(nli_dict):
    sns.set_context("paper", font_scale=1.0)   
    nli_datasets            = ['hans', 'taxinli']
    models                  = ['bertbc-adapter-895K', 'bertbc-lora-592K', 'bertbc-plain', 'bertlc-adapter-3.17M', 'bertlc-lora-1.58M', 'bertlc-plain',\
                            'rb-adapter-895K', 'rb-lora-1.18M', 'rb-plain', 'rl-adapter-3.17M', 'rl-lora-2.63M', 'rl-plain',\
                            'optb-lora-1.57M', 'optb-adapter', 'optb-plain', 'optl-lora-3.15M', 'optl-adapter', 'optl-plain',\
                            't5b-adapter-1.79M', 't5b-lora-1.77M', 't5b-plain', 't5l-adapter-6.34M', 't5l-lora-4.72M', 't5l-plain']

    taxinli_feat_dict   = {
        'lexical_linguistic': 'lexical',
        'factivity_linguistic': 'factivity',
        'syntactic_linguistic': 'syntactic',
        'world_knowledge': 'world_know',
        'taxonomic_knowledge': 'tax_know',
        'conditional_logic': 'connectives',
        'comparative_logic': 'connectives',
        'quantifier_logic':'connectives',
        'negation_logic':'connectives',
        'boolean_logic':'connectives',
        'spatial_reasoning': 'deduction',
        'temporal_reasoning': 'deduction',
        'relational_reasoning': 'deduction',
        'causal_reasoning':'deduction',
        'coreference_reasoning': 'deduction'
    }
    

    hans_heuristic_dict         = ddict(list)
    hans_subcase_dict           = ddict(list)
    taxinli_dict                = ddict(list)
    
    hans_heuristic_corr_dict    = ddict(dict)
    hans_subcases_corr_dict     = ddict(dict)
    taxinli_feature_corr_dict   = ddict(dict)
    
    
    for model in models:
        curr_df                 = nli_dict['hans'][model]
        acc_list                = []
        heuristics              = list(set(curr_df['heuristic']))
        subcases                = list(set(curr_df['subcase']))
        hans_heuristic_dict['model'].append(model)
        hans_subcase_dict['model'].append(model)
        
        for feat in heuristics:
            temp_df                 = curr_df[curr_df['heuristic'] == feat]
            gold_lbls , pred_lbls   = temp_df['label'], temp_df['predictions']
            hans_heuristic_dict[feat].append(round(100*accuracy_score(gold_lbls, pred_lbls),2))
            hans_heuristic_corr_dict[feat][model] = accuracy_score(gold_lbls, pred_lbls)
        
        for feat in subcases:
            temp_df                 = curr_df[curr_df['subcase'] == feat]
            gold_lbls , pred_lbls   = temp_df['label'], temp_df['predictions']
            hans_subcase_dict[feat].append(round(100*accuracy_score(gold_lbls, pred_lbls),2))
            hans_subcases_corr_dict[feat][model] = accuracy_score(gold_lbls, pred_lbls)

    for model in models:
        curr_df                 = nli_dict['taxinli'][model]
        curr_df['feature']      = [taxinli_feat_dict[feat] for feat in curr_df['feature']]
        acc_list                = []
        
        features                = ['lexical', 'factivity', 'syntactic', 'world_know', 'tax_know', 'connectives', 'deduction']
        
        taxinli_dict['model'].append(model)
        for feat in features:
            temp_df                 = curr_df[curr_df['feature'] == feat]
            gold_lbls , pred_lbls   = temp_df['label'], temp_df['predictions']
            taxinli_dict[feat].append(round(100*accuracy_score(gold_lbls, pred_lbls),2))
            taxinli_feature_corr_dict[feat][model] = accuracy_score(gold_lbls, pred_lbls)
                    
    taxinli_df = pd.DataFrame(taxinli_dict)
    taxinli_df.to_csv(f'../out_files/{args.task}_taxinli_features.csv', index=False)
    
    hans_heuristic_df = pd.DataFrame(hans_heuristic_dict)
    hans_heuristic_df.to_csv(f'../out_files/{args.task}_hans_heuristics.csv', index=False)
    
    hans_subcase_df = pd.DataFrame(hans_subcase_dict)
    hans_subcase_df.to_csv(f'../out_files/{args.task}_hans_subcases.csv', index=False)
    
    #### use the HANS dataset  #####
    
    inp_cats            = ['hans_hueristics', 'taxinli_features']
    inp_dicts           = [hans_heuristic_corr_dict, taxinli_feature_corr_dict]
    
    
    
    for inp_cat, inp_dict in zip(inp_cats, inp_dicts):
        corr_dict       = ddict(list)
        feats           = list(inp_dict.keys())
        
        print(feats)
        
        for i, feat1 in enumerate(feats):
            for j, feat2 in enumerate(feats):
                
                src_accs, tgt_accs = [], []
                
                for model in inp_dict[feat1]:
                    src_accs.append(inp_dict[feat1][model])
                    tgt_accs.append(inp_dict[feat2][model])
                
                
                corr_dict['feature1'].append(feat1)
                corr_dict['feature2'].append(feat2)
                corr_dict['corr'].append(pearsonr(src_accs, tgt_accs)[0])
                corr_dict['pvalue'].append(pearsonr(src_accs, tgt_accs)[1])
                
        corr_df = pd.DataFrame(corr_dict)
        sns.heatmap(corr_df.pivot(index='feature1', columns='feature2', values='corr'), cmap='crest', annot=True, fmt=".2f")
        plt.yticks(rotation=0)
        plt.savefig(f'../out_files/{args.task}_{inp_cat}_feat_pr.pdf', facecolor='w', bbox_inches="tight")  
        plt.clf()
        plt.close()
        
    

def compute_sig_change_stances():
    
    sig_df = pd.read_csv(f'../out_files/{args.task}_sig_test.csv')
    
    init_better         = {}
    init_same           = {}
    
    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        if dataset != 'snli': continue
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
        if p_val <= 0.05:
            init_better[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}
        else:
            init_same[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}


    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        
        if dataset in [ 'snli', 'nmonli']: continue
        
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
        if p_val <= 0.05:
            
            if (better_model, worse_model) in init_better:
                init_better[(better_model, worse_model)]['better'] += 1
            elif (worse_model, better_model) in init_better:
                init_better[(worse_model, better_model)]['worse'] += 1
                
                print(f'{better_model} is now better than {worse_model} for {dataset}')
                
            
            elif (better_model, worse_model) in init_same:
                init_same[(better_model, worse_model)]['better'] += 1
            elif (worse_model, better_model) in init_same:
                init_same[(worse_model, better_model)]['worse'] += 1
                # print(f'{worse_model} is now better than {better_model} for {dataset}')
                
        else:
            if (better_model, worse_model) in init_better:
                init_better[(better_model, worse_model)]['equal'] += 1
            elif (worse_model, better_model) in init_better:
                init_better[(worse_model, better_model)]['equal'] += 1
            
            elif (better_model, worse_model) in init_same:
                init_same[(better_model, worse_model)]['equal'] += 1
            elif (worse_model, better_model) in init_same:
                init_same[(worse_model, better_model)]['equal'] += 1

    init_better_cnt = ddict(int)
    init_same_cnt   = ddict(int)
    
    for model_pair in init_better:
        init_better_cnt['better'] += init_better[model_pair]['better']
        init_better_cnt['worse'] += init_better[model_pair]['worse']
        init_better_cnt['equal'] += init_better[model_pair]['equal']
    
    for model_pair in init_same:
        init_same_cnt['better'] += init_same[model_pair]['better']
        init_same_cnt['worse'] += init_same[model_pair]['worse']
        init_same_cnt['equal'] += init_same[model_pair]['equal']    

    pprint(init_better_cnt)
    pprint(init_same_cnt)
    
    




def analyse_sig_test():
    
    sig_df = pd.read_csv(f'../out_files/{args.task}_sig_test.csv')
    
    analysis_cnt_dict = ddict(lambda: ddict(int))
    
    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
        if p_val <= 0.05:
            analysis_cnt_dict[dataset][better_model]    += 1
            analysis_cnt_dict[dataset][worse_model]     -= 1
        
    rank_datasets           = ddict(list)
    
    rank_models_dict        = ddict(list)
    
    
    
    for dataset in analysis_cnt_dict:
        for model in sorted(analysis_cnt_dict[dataset]):
            rank_datasets[dataset].append(analysis_cnt_dict[dataset][model])

        # print the sorted list of models for each dataset
        
        print(f'{dataset} {sorted(analysis_cnt_dict[dataset].items(), key=lambda x: x[1], reverse=True)}', end='\r')
        
        models = sorted(analysis_cnt_dict[dataset].items(), key=lambda x: x[1], reverse=True)
        
        for model in models:
            model_name = model[0]
            start_name = model_name.split('-')[0]
            if 'adapter' in model_name:
                model_name = f'{start_name}-adap'
            elif 'lora' in model_name:
                model_name = f'{start_name}-lora'
            else:
                model_name = start_name
            rank_models_dict[dataset].append(model_name)
    
    rank_models_df = pd.DataFrame(rank_models_dict)
    rank_models_df.to_csv(f'../out_files/{args.task}_rank_models.csv', index=False)        
    
    print('Spearman Rank correlation')
    
    sr_dict = ddict(list)
    kt_dict = ddict(list)
            
    for i, dataset1 in enumerate(rank_datasets):
        for j, dataset2 in enumerate(rank_datasets):
            if dataset1 == 'nmonli' or dataset2 == 'nmonli': continue
            
            sr_dict['model1'].append(dataset1)
            sr_dict['model2'].append(dataset2)
            sr_dict['corr'].append(spearmanr(rank_datasets[dataset1], rank_datasets[dataset2])[0])
            sr_dict['pvalue'].append(spearmanr(rank_datasets[dataset1], rank_datasets[dataset2])[1])
            
            kt_dict['model1'].append(dataset1)
            kt_dict['model2'].append(dataset2)
            kt_dict['corr'].append(kendalltau(rank_datasets[dataset1], rank_datasets[dataset2])[0])
            kt_dict['pvalue'].append(kendalltau(rank_datasets[dataset1], rank_datasets[dataset2])[1])
    
            print(dataset1, dataset2,  spearmanr(rank_datasets[dataset1], rank_datasets[dataset2])[0])

    sr_df = pd.DataFrame(sr_dict)
    kt_df = pd.DataFrame(kt_dict)
    
    p_cnts = 0
    for idx, row in sr_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(sr_df)}')  
    
    p_cnts = 0
    for idx, row in kt_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    
    # sns.heatmap(sr_df.pivot(index='model1', columns='model2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    # plt.savefig('../out_files/sr_rank.pdf', facecolor='w', bbox_inches="tight")
    
    sns.heatmap(kt_df.pivot(index='model1', columns='model2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig('../out_files/kt_rank.pdf', facecolor='w', bbox_inches="tight")
    
    
def validate():
    
    sig_df = pd.read_csv(f'{args.task}_sig_test.csv')
    mis_cnts = ddict(lambda: ddict(int))
    
    
    for index, row in sig_df.iterrows():
        
        if row['worse_acc'] > row['better_acc']:
            mis_cnts[row['dataset']][row['worse_acc']- row['better_acc']] += 1
    
    pprint(mis_cnts)

### Population Wise metrics

def get_all_population_metrics(scores):
    """
    (a) central tendency: arithmetic mean, geometric mean, median, midhinge.
    (b) variability: variance, minimum, maximum, range, interquartile range.
    (c) shape: skewness, kurtosis
    """
    assert len(scores) > 0
    
    score_dict = {}
    score_dict['AM'] = np.mean(scores)
    score_dict['GM'] = np.power(np.prod(scores), 1/len(scores))
    score_dict['median'] = np.median(scores)
    score_dict['Q1'] = np.percentile(scores, 25) 
    score_dict['Q3'] = np.percentile(scores, 75) 
    score_dict['midhinge'] = (score_dict['Q1'] + score_dict['Q3'])/2
    
    score_dict['var']   = np.var(scores)
    score_dict['min']   = np.min(scores)
    score_dict['max']   = np.max(scores)
    score_dict['range'] = score_dict['max'] - score_dict['min']
    score_dict['IQR']   = score_dict['Q3'] - score_dict['Q1']
    
    score_dict['skew']      = scipy.stats.skew(scores)
    score_dict['kurtosis']  = scipy.stats.kurtosis(scores)
    
    
    return score_dict



def compute_population_metrics(nli_dict):
    
    sns.set_theme(style='whitegrid')
    sns.set_context("paper", font_scale=1.0)   
    
    
    dataset_stats         = ddict(list)
    model_stats           = ddict(list)
    

    for dataset in nli_dict:
        if dataset == 'nmonli': 
            continue
        
        for model in nli_dict[dataset]:
            curr_df = nli_dict[dataset][model]
            
            gold_lbls , pred_lbls = curr_df['label'], curr_df['predictions']
            scores                = [1 if i==j else 0 for i,j in zip(gold_lbls, pred_lbls)]
            
            score_dict            = get_all_population_metrics(scores)
            
            dataset_stats['model'].append(model)
            dataset_stats['dataset'].append(dataset)
            
            
            for key in score_dict:
                dataset_stats[key].append(round(score_dict[key],2))
            
            dataset_stats['num_samples'].append(len(scores))
            dataset_stats['accuracy'].append(round(accuracy_score(gold_lbls, pred_lbls),2))
            dataset_stats['f1'].append(round(f1_score(gold_lbls, pred_lbls, average='macro'),2))
            
    
    dataset_df          = pd.DataFrame(dataset_stats)
    
    dataset_df.to_csv(f'{args.task}_population_dataset_stats.csv', index=False)
    
    models              = list(set(dataset_df['model']))
    
    for model in sorted(models):
        scores                = list(dataset_df[dataset_df['model'] == model]['accuracy'])
        score_dict            = get_all_population_metrics(scores)
        
        model_stats['model'].append(model)
        
        for key in score_dict:
            model_stats[key].append(round(score_dict[key],2))
            
        model_stats['num_samples'].append(len(scores))
    
    ### compute the values for accuracy scores
        
    model_df            = pd.DataFrame(model_stats)
    model_df.to_csv(f'{args.task}_population_model_stats.csv', index=False)
    
    model_median_perf = {}
    for model in models: 
        model_median_perf[model] = np.median(list(dataset_df[dataset_df['model'] == model]['accuracy']))
    
    # return list of models in increasing order of median performance
    model_median_perf = [(k,v) for k, v in sorted(model_median_perf.items(), key=lambda item: item[1])]
    sorted_models     = [k for k,v in model_median_perf]
    
    # plot the box and whisker plot for different models
    
    sns.boxplot(data=dataset_df, x='accuracy', y='model', order=sorted_models)
    plt.tight_layout()
    plt.savefig(f'boxplot.png', dpi=300)
    plt.close()
    
    ### compute the values for f1 scores
    
    model_stats           = ddict(list)
    
    for model in sorted(models):
        scores                = list(dataset_df[dataset_df['model'] == model]['f1'])
        score_dict            = get_all_population_metrics(scores)
        
        model_stats['model'].append(model)
        
        for key in score_dict:
            model_stats[key].append(round(score_dict[key],2))
            
        model_stats['num_samples'].append(len(scores))
        
        
    model_df            = pd.DataFrame(model_stats)
    model_df.to_csv(f'{args.task}_population_model_f1_stats.csv', index=False)
    
    model_median_perf = {}
    for model in models: 
        model_median_perf[model] = np.median(list(dataset_df[dataset_df['model'] == model]['f1']))
    
    # return list of models in increasing order of median performance
    model_median_perf = [(k,v) for k, v in sorted(model_median_perf.items(), key=lambda item: item[1])]
    sorted_models     = [k for k,v in model_median_perf]
    
    # plot the box and whisker plot for different models
    
    sns.boxplot(data=dataset_df, x='f1', y='model', order=sorted_models)
    plt.tight_layout()
    plt.savefig(f'f1_boxplot.png', dpi=300)
    plt.close()
    

    
# compute architecture wise correlations
def compute_archwise_correlations(nli_dict):
    
    nli_named_dict = {
        'snli'      : 'SNLI',
        'mnli'      : 'MNLI-mm',
        'matmnli'   : 'MNLI-m',
        'hans'      : 'HANS',
        'snli-cf'   : 'SNLI-CF',
        'conjnli'   : 'ConjNLI',
        'taxinli'   : 'TaxiNLI',
        'snli-bt'   : 'SNLI-BT',
        'pmonli'    : 'PMoNLI',
        'snli_hard' : 'SNLI-H',
    }

    results_dict          = ddict(list) 

    # nli_datasets          = ['snli',  'mnli', 'snli-cf', 'conjnli', 'matmnli', 'hans', 'taxinli', 'snli-bt', 'pmonli', 'snli_hard']
    nli_datasets          = ['snli',  'matmnli', 'mnli', 'taxinli', 'snli-bt', 'snli-cf', 'snli_hard', 'hans',  'conjnli',    'pmonli']
    models                = ['bertbc-adapter-895K', 'bertbc-lora-592K', 'bertbc-plain', 'bertlc-adapter-3.17M', 'bertlc-lora-1.58M', 'bertlc-plain',\
                            'rb-adapter-895K', 'rb-lora-1.18M', 'rb-plain', 'rl-adapter-3.17M', 'rl-lora-2.63M', 'rl-plain',\
                            'optb-adapter', 'optb-lora-1.57M', 'optb-plain', 'optl-adapter', 'optl-lora-3.15M', 'optl-plain',\
                            't5b-adapter-1.79M', 't5b-lora-1.77M', 't5b-plain', 't5l-adapter-6.34M', 't5l-lora-4.72M', 't5l-plain']

    acc_dict              = ddict(dict)
    datasetwise_dict      = ddict(list)
    
    for model in models:
        results_dict['Model'].append(model)
        
        for dataset in nli_datasets:
            curr_df                     = nli_dict[dataset][model]
            acc_list                    = []
            for seed in ['1', '2', '3']:
                temp_df                 = curr_df[curr_df['seed'] == seed]
                gold_lbls , pred_lbls   = temp_df['label'], temp_df['predictions']
                acc                     = accuracy_score(gold_lbls, pred_lbls)
                acc_list.append(acc)
                datasetwise_dict[dataset].append(acc)
            
            mean_acc                    = round(100*np.mean(acc_list),1)
            std_acc                     = round(100*np.std(acc_list),1)
            
            results_dict[dataset].append(f'{mean_acc}\pm{std_acc}')            
            acc_dict[dataset][model] = mean_acc
    
    results_dict['Model'].append('Avg')    
    for dataset in datasetwise_dict:
        accs = datasetwise_dict[dataset]
        mean_acc                    = round(100*np.mean(accs),1)
        std_acc                     = round(100*np.std(accs),1)
        results_dict[dataset].append(f'{mean_acc}\pm{std_acc}')
        
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'../out_files/{args.task}_datasetwise_metrics.csv', index=False)
    
    corr_dict    = ddict(list)
    sr_dict = ddict(list)
    kt_dict = ddict(list)
    pr_dict = ddict(list)
            
    for i, dataset1 in enumerate(nli_datasets):
        for j, dataset2 in enumerate(nli_datasets):
            
            src_accs, tgt_accs = [], []
            
            for model in acc_dict[dataset1]:
                src_accs.append(acc_dict[dataset1][model])
                tgt_accs.append(acc_dict[dataset2][model])                
            
            sr_dict['dataset1'].append(nli_named_dict[dataset1])
            sr_dict['dataset2'].append(nli_named_dict[dataset2])
            sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
            sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
            
            kt_dict['dataset1'].append(nli_named_dict[dataset1])
            kt_dict['dataset2'].append(nli_named_dict[dataset2])
            kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
            kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
            
            pr_dict['dataset1'].append(nli_named_dict[dataset1])
            pr_dict['dataset2'].append(nli_named_dict[dataset2])
            pr_dict['corr'].append(pearsonr(src_accs, tgt_accs)[0])
            pr_dict['pvalue'].append(pearsonr(src_accs, tgt_accs)[1])
            
            print(dataset1, dataset2, spearmanr(src_accs, tgt_accs)[0], kendalltau(src_accs, tgt_accs)[0])

    sr_df = pd.DataFrame(sr_dict)
    kt_df = pd.DataFrame(kt_dict)
    pr_df = pd.DataFrame(pr_dict)
    
    p_cnts = 0
    for idx, row in sr_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(sr_df)}')  
    
    p_cnts = 0
    for idx, row in kt_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(kt_df)}')  
    
    p_cnts = 0
    for idx, row in pr_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(pr_df)}')  
    
    
    sns.heatmap(sr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig('../out_files/nli_archwise_sr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    # how to clear the matplotlib.pyplot 
    sns.heatmap(kt_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig('../out_files/nli_archwise_kt_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    sns.heatmap(pr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig('../out_files/nli_archwise_pr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    # once we have the rankings for each dataset; obtain the global ranks for each dataset using Borda counts
    
    rank_dict = ddict(dict)
    
    for dataset in acc_dict:
        if dataset in ['nmonli']: continue
        
        accs = acc_dict[dataset]
        
        # sort the dictionary accs in reverse order of items 
        model_ranks = [(k,v) for k,v in sorted(accs.items(), key=lambda item: item[1], reverse=True)]
                
        for idx, model_rank in enumerate(model_ranks):
            rank_dict[model_rank[0]][dataset] = idx

    modelwise_rank_dict = ddict(list)
    
    for models in rank_dict:
        modelwise_rank_dict['model'].append(models)
        for dataset in rank_dict[models]:
            modelwise_rank_dict[dataset].append(rank_dict[models][dataset])
            
    rank_df                      = pd.DataFrame.from_dict(modelwise_rank_dict)
    rank_csv_file                = '../out_files/archwise_ranks.csv'    
    rank_df.to_csv(rank_csv_file, index = False)

    aggregated_ranks             = mc4_aggregator(rank_csv_file, header_row = 0, index_col = 0)
    
    with open('../out_files/nli_archwise_aggregated_ranks.json', 'w') as wf:
        json.dump(aggregated_ranks, wf, indent=4)

# compute instance wise correlations
def compute_instancewise_correlations():
    
    nli_named_dict = {
        'snli'      : 'SNLI',
        'mnli'      : 'MNLI-mm',
        'matmnli'   : 'MNLI-m',
        'hans'      : 'HANS',
        'snli-cf'   : 'SNLI-CF',
        'conjnli'   : 'ConjNLI',
        'taxinli'   : 'TaxiNLI',
        'snli-bt'   : 'SNLI-BT',
        'pmonli'    : 'PMoNLI',
        'snli_hard' : 'SNLI-H'
    }
    
    with open('../out_files/nli_instancewise_results.json', 'r') as f:
        acc_dict = json.load(f)
        
    nli_datasets = ['snli','mnli', 'snli-cf', 'conjnli', 'matmnli', 'hans', 'taxinli', 'snli-bt', 'pmonli', 'snli_hard']
    
    corr_dict    = ddict(list)
    
    
    sr_dict = ddict(list)
    kt_dict = ddict(list)
    pr_dict = ddict(list)
            
    for i, dataset1 in enumerate(nli_datasets):
        for j, dataset2 in enumerate(nli_datasets):
            
            src_accs, tgt_accs = [], []
            
            for model in acc_dict[dataset1]:
                src_accs.append(acc_dict[dataset1][model])
                tgt_accs.append(acc_dict[dataset2][model])                
            
            sr_dict['dataset1'].append(nli_named_dict[dataset1])
            sr_dict['dataset2'].append(nli_named_dict[dataset2])
            sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
            sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
            
            kt_dict['dataset1'].append(nli_named_dict[dataset1])
            kt_dict['dataset2'].append(nli_named_dict[dataset2])
            kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
            kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
            
            pr_dict['dataset1'].append(nli_named_dict[dataset1])
            pr_dict['dataset2'].append(nli_named_dict[dataset2])
            pr_dict['corr'].append(pearsonr(src_accs, tgt_accs)[0])
            pr_dict['pvalue'].append(pearsonr(src_accs, tgt_accs)[1])
            
            print(dataset1, dataset2, spearmanr(src_accs, tgt_accs)[0], kendalltau(src_accs, tgt_accs)[0], len(src_accs))

    
    sr_df = pd.DataFrame(sr_dict)
    kt_df = pd.DataFrame(kt_dict)
    pr_df = pd.DataFrame(pr_dict)
    
    p_cnts = 0
    for idx, row in sr_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(sr_df)}')  
    
    p_cnts = 0
    for idx, row in kt_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(kt_df)}')  
    
    p_cnts = 0
    for idx, row in pr_df.iterrows():
        if row['pvalue'] < 0.05:
            p_cnts += 1
    
    print(f'Number of significant correlations: {p_cnts}/ {len(pr_df)}')  
    
    
    sns.heatmap(sr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig(f'../out_files/{args.task}_instwise_sr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    sns.heatmap(kt_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig(f'../out_files/{args.task}_instwise_kt_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    sns.heatmap(pr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.savefig(f'../out_files/{args.task}_instwise_pr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    
    
    # once we have the rankings for each dataset; obtain the global ranks for each dataset using Borda counts
    
    rank_dict = ddict(dict)
    
    for dataset in acc_dict:
        if dataset in ['nmonli']: continue
        
        accs = acc_dict[dataset]
        
        # sort the dictionary accs in reverse order of items 
        model_ranks = [(k,v) for k,v in sorted(accs.items(), key=lambda item: item[1], reverse=True)]
                
        for idx, model_rank in enumerate(model_ranks):
            rank_dict[model_rank[0]][dataset] = idx

    modelwise_rank_dict = ddict(list)
    
    for models in rank_dict:
        modelwise_rank_dict['model'].append(models)
        for dataset in rank_dict[models]:
            modelwise_rank_dict[dataset].append(rank_dict[models][dataset])
            
    rank_df                      = pd.DataFrame.from_dict(modelwise_rank_dict)
    rank_csv_file                = '../out_files/instwise_ranks.csv'    
    rank_df.to_csv(rank_csv_file, index = False)

    aggregated_ranks             = mc4_aggregator(rank_csv_file, header_row = 0, index_col = 0)
    
    with open('../out_files/nli_instwise_aggregated_ranks.json', 'w') as wf:
        json.dump(aggregated_ranks, wf, indent=4)
        # agg_ranks_df.to_csv('../out_files/nli_instwise_aggregated_ranks.csv', index=False)
    

def cross_task_corr():
    
    with open('../out_files/nli_instwise_aggregated_ranks.json', 'r') as wf:
        agg_nli_dict = json.load(wf)
        
    with open('../../mrc/results/mrc_instwise_aggregated_ranks.json', 'r') as wf:
        agg_mrc_dict = json.load(wf)
    
    nli_models = list(agg_nli_dict.keys())
    mrc_models = list(agg_mrc_dict.keys())
    
    print(sorted(nli_models))
    print(sorted(mrc_models))
    
    
    nli_scores = [agg_nli_dict[model] for model in sorted(nli_models)]
    mrc_scores = [agg_mrc_dict[model] for model in sorted(mrc_models)]
    
    print(spearmanr(nli_scores, mrc_scores))

    with open('../out_files/nli_archwise_aggregated_ranks.json', 'r') as wf:
        agg_nli_dict = json.load(wf)
        
    with open('../../mrc/results/mrc_archwise_aggregated_ranks.json', 'r') as wf:
        agg_mrc_dict = json.load(wf)
    
    nli_models = list(agg_nli_dict.keys())
    mrc_models = list(agg_mrc_dict.keys())
    
    print(sorted(nli_models))
    print(sorted(mrc_models))
    
    
    nli_scores = [agg_nli_dict[model] for model in sorted(nli_models)]
    mrc_scores = [agg_mrc_dict[model] for model in sorted(mrc_models)]
    
    print(spearmanr(nli_scores, mrc_scores))

def get_model_cat(model):
    ft_type = 'plain'
    size_type = 'base'
    arch_type = 'eo'
    
    if 'adapter' in model:
        ft_type = 'adapter'
    elif 'lora' in model:
        ft_type = 'lora'
    else:
        ft_type = 'plain'

    if model.startswith('t5'):
        arch_type = 'ed'
    elif model.startswith('opt'):
        arch_type = 'do'
    else:
        arch_type = 'eo'
    
    if 'l-' in model or 'lc-' in model:
        size_type = 'large'
    
    return {'ft': ft_type, 'size': size_type, 'arch': arch_type}

def comp_anova():
    
    gen_type_dict = {
        'mnli'      : 'OOD',
        'matmnli'   : 'OOD',
        'hans'      : 'ROB',
        'snli-cf'   : 'ROB',
        'conjnli'   : 'Comp',
        'taxinli'   : 'OOD',
        'snli-bt'   : 'ROB',
        'pmonli'    : 'Comp',
        'snli_hard' : 'ROB',
        'snli'      : 'A',
    }
    
    
    with open('../out_files/nli_instancewise_results.json', 'r') as f:
        acc_dict = json.load(f)
        
    nli_datasets = ['snli', 'mnli', 'snli-cf', 'conjnli', 'matmnli', 'hans', 'taxinli', 'snli-bt', 'snli_hard', 'pmonli']
    
    results_dict     = ddict(list)
            
    for i, dataset in enumerate(nli_datasets):
        for model in acc_dict[dataset]:
            
            tgt_acc = acc_dict[dataset][model]
            src_acc = acc_dict['snli'][model]
            
            results_dict['tgt_acc'].append(tgt_acc)
            results_dict['src_acc'].append(src_acc)
            results_dict['Gen_type'].append(gen_type_dict[dataset])
            results_dict['NSD'].append((tgt_acc- src_acc)/src_acc)
            
            model_cat = get_model_cat(model)
            results_dict['arch'].append(model_cat['arch'])
            results_dict['ft'].append(model_cat['ft'])
            results_dict['size'].append(model_cat['size'])
    
    results_df      = pd.DataFrame(results_dict)
    results_df.to_csv(f'../out_files/{args.task}_acc_results.csv', index=False)
    
    # how to save the tgt_acc in a sns.plot ##
    
    sns.histplot(data=results_df, x='tgt_acc', kde=True, bins=50)
    plt.savefig('../out_files/tgt_acc_hist.pdf', facecolor='w', bbox_inches="tight")
    
    
    
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols 
    
    
    reg_model = ols('tgt_acc ~ C(Gen_type) + C(arch) + C(ft) + C(size)', data=results_df).fit()
    anova_table = sm.stats.anova_lm(reg_model, typ=2)
    print(anova_table)    
    print(reg_model.params)
    
    reg_model = ols('NSD ~ C(Gen_type) + C(arch) + C(ft) + C(size)', data=results_df).fit()
    anova_table = sm.stats.anova_lm(reg_model, typ=2)
    print(anova_table)    
    print(reg_model.params)
    
    # reg_model = ols('NSD ~ C(Gen_type)* C(arch) * C(ft) * C(size)', data=results_df).fit()
    # anova_table = sm.stats.anova_lm(reg_model, typ=2)
    # print(anova_table)    
    # print(reg_model.params)
    
    
    # model = ols('tgt_acc ~ C(Gen_type) + C(arch) + C(ft) + C(size) +  src_acc + \
    #     C(Gen_type):C(arch) + C(Gen_type):C(ft)+ C(Gen_type):C(size)+ C(ft):C(arch) + C(size):C(arch) + C(ft):C(size)+ \
    #     C(Gen_type):C(arch):C(ft) + C(Gen_type):C(arch):C(size) + C(Gen_type):C(size):C(ft) + C(size):C(arch):C(ft)+ \
    #     C(Gen_type):C(arch):C(ft):C(size)', data=results_df).fit()

    # model = ols('tgt_acc ~ C(Gen_type)*C(arch)*C(ft)*C(size)*src_acc', data=results_df).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)    
    # print(model.params)    
    
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute significance test')
    parser.add_argument('--task',       type=str, default='nli', help='task')
    parser.add_argument('--inp_dir',    type=str, default='../prediction_files', help='input directory of predictions')
    parser.add_argument('--step',       type=str, default='analyse', required=True)
    
    args        =  parser.parse_args()
    
    if args.step == 'create':
        nli_dict    =  create_nli_dict()
        with open('../out_files/nli_dict.pkl', 'wb') as wf:
            pickle.dump(nli_dict, wf)
        
    if args.step == 'sig_test':
        with open('../out_files/nli_dict.pkl', 'rb') as wf:
            nli_dict = pickle.load(wf)
            
        perform_sig_test(nli_dict)
    
    if args.step == 'analyse':
        analyse_sig_test()
        
    if args.step == 'pop':
        
        with open('nli_dict.pkl', 'rb') as wf:
            nli_dict = pickle.load(wf)
        
        compute_population_metrics(nli_dict)
        
    if args.step == 'validate_hyps':

        validate_sig_hyps()

    if args.step == 'data_res':
        compute_datasetwise_results()

        
    if args.step == 'inst_corr':
        
        compute_instancewise_correlations()
    
    
        
    if args.step == 'arch_corr':
        with open('../out_files/nli_dict.pkl', 'rb') as wf:
            nli_dict = pickle.load(wf)
        
        compute_archwise_correlations(nli_dict)
        
        
    if args.step == 'inp_cats':
        with open('../out_files/nli_dict.pkl', 'rb') as wf:
            nli_dict = pickle.load(wf)
        
        get_input_categories(nli_dict)
        
    if args.step == 'cross_task':
        cross_task_corr()
    
    if args.step == 'change_stance':
        compute_sig_change_stances()
        
    if args.step == 'anova':
        comp_anova()
    
    # parser.add_argument('file1', type=str, help='file1')
    # parser.add_argument('file2', type=str, help='file2')
    # parser.add_argument('model1', type=str, help='model1')
    # parser.add_argument('model2', type=str, help='model2')
    # parser.add_argument('num_iter', type=int, help='num_iter')
    # args = parser.parse_args()
    
    # file1 = args.file1
    # file2 = args.file2
    # model1 = args.model1
    # model2 = args.model2
    # num_iter = args.num_iter
    
    # df1 = pd.read_csv(file1)
    # df2 = pd.read_csv(file2)
    
    # sig_val, p_val = compute_bootstrapped_df(df1, df2, model1, model2, num_iter)
    
    # print(f"{model1} vs {model2} p-value: {p_val} {sig_val}")
    
    # sys.exit(0)
    
    #print(f"p-value: {p_val} {sig_val}")
    #print(f"{model1} vs {model2}