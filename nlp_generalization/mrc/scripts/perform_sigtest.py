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
from scipy.stats import spearmanr, kendalltau, pearsonr
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import json
from evaluate import load
from mc4.algorithm import mc4_aggregator



def compute_sig_change_stances():
    
    sig_df = pd.read_csv(f'../results/{args.task}_sig_test.csv')
    
    init_better         = {}
    init_same           = {}
    
    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        if dataset != 'squad': continue
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
        if p_val <= 0.05:
            init_better[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}
        else:
            init_same[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}


    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        
        if dataset == 'squad': continue
        
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
    



def compute_bootstrapped_df(df, model1, model2, num_iter = 100):
        
    metric   = load("squad")
    idx_list = [idx for idx in range(len(df))]
    
    cnt        = 0
    
    gold_model1   = df[f'{model1}_references'].tolist()
    gold_model2   = df[f'{model2}_references'].tolist()
    
    pred_model1   = df[f'{model1}_predictions'].tolist()
    pred_model2   = df[f'{model2}_predictions'].tolist()
    
    model1_f1     = metric.compute(references= gold_model1, predictions= pred_model1)['f1']
    model2_f1     = metric.compute(references= gold_model2, predictions= pred_model2)['f1']

        
    sys_diff      = model1_f1 - model2_f1
    
    if sys_diff    <= 0:        
        temp_model = model1
        model1     = model2
        model2     = temp_model
        temp_f1    = model1_f1
        model1_f1  = model2_f1
        model2_f1  = temp_f1
        sys_diff   = -sys_diff    
    
    assert sys_diff > 0
    
    for i in range(0, num_iter):
        random.seed(i)
        np.random.seed(i)
        curr_idx_list = random.choices(idx_list, k=min(int(0.1*len(idx_list)), 1000))
        
        pred_model1 = df.iloc[curr_idx_list][f'{model1}_predictions'].tolist()
        pred_model2 = df.iloc[curr_idx_list][f'{model2}_predictions'].tolist()
        
        gold_model1 = df.iloc[curr_idx_list][f'{model1}_references'].tolist()
        gold_model2 = df.iloc[curr_idx_list][f'{model2}_references'].tolist()
         
        curr_model1_f1   = metric.compute(references= gold_model1, predictions= pred_model1)['f1']
        curr_model2_f1   = metric.compute(references= gold_model2, predictions= pred_model2)['f1']
        
        # comes from the 2012 non-paramteric method of boostrap significance testing
        curr_diff        = curr_model1_f1 - curr_model2_f1
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
        'better_model': model1,
        'better_f1': round(model1_f1,2),
        'worse_model': model2,
        'worse_f1': round(model2_f1,2)
    }
    
    return results_dict



def create_mrc_dict():
    mrc_dict = ddict(lambda: ddict(lambda: ddict(list)))
    dataset  = None
    
    for file in tqdm(os.listdir(args.inp_dir), desc='Reading files'):
        assert file.endswith('.json')

        dataset        = file.split('-')[0]
        model_name     = "-".join(file.split('-')[1:3])
        seed           = file.split('-')[-1].replace('.json', '')
        
        with open(os.path.join(args.inp_dir, file), 'r') as f:
            data = json.load(f)
        
        for ex in data:
            answers = data[ex]["label_text"]
            curr_answers = {'answer_start': [], 'text': []}
            
            for ans in answers:
                curr_answers['answer_start'].append(0)
                curr_answers['text'].append(ans)
            
            
            mrc_dict[dataset][model_name]['id'].append(f'{ex}-{seed}')
            mrc_dict[dataset][model_name][f'{model_name}_references'].append({"id": f'{ex}-{seed}', "answers": curr_answers})
            mrc_dict[dataset][model_name][f'{model_name}_predictions'].append({"id": f'{ex}-{seed}', "prediction_text": data[ex]["prediction_text"]})
        
        
    with open(f'../results/mrc_dict.json', 'w') as f:
        json.dump(mrc_dict, f, indent=4)



def create_instancewise_dict():
    mrc_dict = ddict(dict)
    dataset  = None
    metric   = load("squad")
    
    for file in tqdm(os.listdir(args.inp_dir), desc='Reading files'):
        assert file.endswith('.json')

        dataset        = file.split('-')[0]
        model_name     = "-".join(file.split('-')[1:3])
        seed           = file.split('-')[-1].replace('.json', '')
        
        with open(os.path.join(args.inp_dir, file), 'r') as f:
            data = json.load(f)
        
        refs, preds    = [], []
        
        for ex in data:
            answers         = data[ex]["label_text"]
            curr_answers    = {'answer_start': [], 'text': []}
            
            for ans in answers:
                curr_answers['answer_start'].append(0)
                curr_answers['text'].append(ans)
            
            refs.append({"id": f'{ex}', "answers": curr_answers})
            preds.append({"id": f'{ex}', "prediction_text": data[ex]["prediction_text"]})
        
        f1     = metric.compute(references= refs, predictions= preds)['f1']
        
        mrc_dict[dataset][f'{model_name}-{seed}'] = f1

    with open(f'../results/instancewise_mrc_dict.json', 'w') as f:
        json.dump(mrc_dict, f, indent=4)


def get_dataset_name(dataset):
    
    if dataset == 'adversarial_hotpotqa':
        return 'adv_HQA'
    elif dataset == 'adversarial_squad':
        return 'adv_squad'
    else:
        return dataset
    

def create_instancewise_corr():
    mrc_named_dict = {
        'newsqa': 'NewsQA',
        'adversarial_squad': 'Adv-SQuAD',
        'adversarial_hotpotqa': 'Adv-HPQA',
        'squad': 'SQuAD',
        'musique': 'MusiQuE'
    }
    
    with open(f'../results/instancewise_mrc_dict.json', 'r') as f:
        mrc_dict = json.load(f)
        
    mrc_datasets = mrc_dict.keys()
    
    corr_dict    = ddict(list)
    
    sr_dict = ddict(list)
    kt_dict = ddict(list)
    pr_dict = ddict(list)
            
    for i, dataset1 in enumerate(mrc_datasets):
        for j, dataset2 in enumerate(mrc_datasets):
            
            src_accs, tgt_accs = [], []
            
            for model in mrc_dict[dataset1]:
                src_accs.append(mrc_dict[dataset1][model])
                tgt_accs.append(mrc_dict[dataset2][model])                
            
                
            sr_dict['dataset1'].append(mrc_named_dict[dataset1])
            sr_dict['dataset2'].append(mrc_named_dict[dataset2])
            sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
            sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
            
            kt_dict['dataset1'].append(mrc_named_dict[dataset1])
            kt_dict['dataset2'].append(mrc_named_dict[dataset2])
            kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
            kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
            
            pr_dict['dataset1'].append(mrc_named_dict[dataset1])
            pr_dict['dataset2'].append(mrc_named_dict[dataset2])
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
    
    
    sns.heatmap(sr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('../results/mrc_instwise_sr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    
    # how to clear the matplotlib.pyplot 
    
    sns.heatmap(kt_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('../results/mrc_instwise_kt_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()

    sns.heatmap(pr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('../results/mrc_instwise_pr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()




    rank_dict = ddict(dict)
    
    for dataset in mrc_dict:
        accs = mrc_dict[dataset]
        
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
    rank_csv_file                = '../results/instwise_ranks.csv'    
    rank_df.to_csv(rank_csv_file, index = False)

    aggregated_ranks             = mc4_aggregator(rank_csv_file, header_row = 0, index_col = 0)
    
    with open('../results/mrc_instwise_aggregated_ranks.json', 'w') as wf:
        json.dump(aggregated_ranks, wf, indent=4)
    


    
def sigtest():
    
    # load the mrc dict
    with open(f'../results/mrc_dict.json', 'r') as f:
        mrc_dict = json.load(f)
    
    mrc_datasets = sorted(list(mrc_dict.keys()))
    mrc_models   = sorted(list(mrc_dict['squad'].keys()))
    
    print(mrc_models)
    sigtest_dict = ddict(list)
    
    for mrc_dataset in mrc_datasets:        
        for idx1, model1 in enumerate(mrc_models):
            for idx2, model2 in enumerate(mrc_models):
                
                if 'optb-adapter' not in model1 and 'optl-adapter' not in model1 and 'optb-adapter' not in model2 and 'optl-adapter' not in model2 :
                    continue
                
                if idx1 <= idx2:
                    continue
                
                df1 = pd.DataFrame(mrc_dict[mrc_dataset][model1])
                df2 = pd.DataFrame(mrc_dict[mrc_dataset][model2])
                
                df  = pd.merge(df1, df2, on='id')
                
                print(f'Done for {model1} vs {model2} for {mrc_dataset}', end= '\r')

                results_dict        = compute_bootstrapped_df(df, model1, model2, num_iter = 1000)
                
                print(f'{mrc_dataset} | {model1} vs {model2} | {results_dict["sig_val"]} | {results_dict["p_val"]} | {results_dict["better_model"]} | {results_dict["better_f1"]} | {results_dict["worse_model"]} | {results_dict["worse_f1"]}')


                sigtest_dict['dataset'].append(mrc_dataset)
                sigtest_dict['model1'].append(model1)
                sigtest_dict['model2'].append(model2)
                sigtest_dict['sig_val'].append(results_dict['sig_val'])
                sigtest_dict['p_val'].append(results_dict['p_val'])
                sigtest_dict['better_model'].append(results_dict['better_model'])
                sigtest_dict['better_f1'].append(results_dict['better_f1'])
                sigtest_dict['worse_model'].append(results_dict['worse_model'])
                sigtest_dict['worse_f1'].append(results_dict['worse_f1'])
                
    sigtest_df = pd.DataFrame(sigtest_dict)            
    sigtest_df.to_csv(f'../results/opt-adapters_sigtest.csv', index=False)




def create_architecturewise_correlations():
    
    mrc_named_dict = {
        'newsqa': 'NewsQA',
        'adversarial_squad': 'Adv-SQuAD',
        'adversarial_hotpotqa': 'Adv-HPQA',
        'squad': 'SQuAD',
        'musique': 'MusiQuE'
    }
    
    metric                = load('squad')
    mrc_dict              = ddict(lambda: ddict(list))
    results_dict          = ddict(list)
    acc_dict              = ddict(dict)
    datasetwise_scores    = ddict(list)
    
    for file in tqdm(os.listdir(args.inp_dir), desc='Reading files'):
        assert file.endswith('.json')

        dataset        = file.split('-')[0]
        model_name     = "-".join(file.split('-')[1:3])
        seed           = file.split('-')[-1].replace('.json', '')
        
        with open(os.path.join(args.inp_dir, file), 'r') as f:
            data = json.load(f)
        
        refs, preds    = [], [] 
        for ex in data:
            answers         = data[ex]["label_text"]
            curr_answers    = {'answer_start': [], 'text': []}
            
            for ans in answers:
                curr_answers['answer_start'].append(0)
                curr_answers['text'].append(ans)
            
            refs.append({"id": f'{ex}', "answers": curr_answers})
            preds.append({"id": f'{ex}', "prediction_text": data[ex]["prediction_text"]})
        
        f1     = metric.compute(references= refs, predictions= preds)['f1']
        mrc_dict[model_name][dataset].append(f1)
        datasetwise_scores[dataset].append(f1)

    for model_name in sorted(mrc_dict):     
        results_dict['Model'].append(model_name)
        for dataset in mrc_dict[model_name]:
            f1s = mrc_dict[model_name][dataset]
            f1_score = f'{round(np.mean(f1s),1)}$\pm${round(np.std(f1s),1)}' 
            results_dict[dataset].append(f1_score)
            acc_dict[dataset][model_name] = np.mean(f1s)
            
                            
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'../results/{args.task}_datasetwise_metrics.csv', index=False)

    for dataset in datasetwise_scores:
        print(dataset, np.mean(datasetwise_scores[dataset]), np.std(datasetwise_scores[dataset]))


    mrc_datasets        = list(acc_dict.keys())                            

    sr_dict = ddict(list)
    kt_dict = ddict(list)
    pr_dict = ddict(list)
            
    for i, dataset1 in enumerate(mrc_datasets):
        for j, dataset2 in enumerate(mrc_datasets):    
            src_accs, tgt_accs = [], []
            
            for model in acc_dict[dataset1]:
                src_accs.append(acc_dict[dataset1][model])
                tgt_accs.append(acc_dict[dataset2][model])                
                
            sr_dict['dataset1'].append(mrc_named_dict[dataset1])
            sr_dict['dataset2'].append(mrc_named_dict[dataset2])
            sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
            sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
            
            kt_dict['dataset1'].append(mrc_named_dict[dataset1])
            kt_dict['dataset2'].append(mrc_named_dict[dataset2])
            kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
            kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
            
            pr_dict['dataset1'].append(mrc_named_dict[dataset1])
            pr_dict['dataset2'].append(mrc_named_dict[dataset2])
            pr_dict['corr'].append(pearsonr(src_accs, tgt_accs)[0])
            pr_dict['pvalue'].append(pearsonr(src_accs, tgt_accs)[1])
            
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
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f'../results/{args.task}_archwise_sr_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    # how to clear the matplotlib.pyplot 
    
    sns.heatmap(kt_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f'../results/{args.task}_archwise_kt_rank.pdf', facecolor='w', bbox_inches="tight")
    plt.clf()
    plt.close()
    # how to clear the matplotlib.pyplot 
    
    sns.heatmap(pr_df.pivot(index='dataset1', columns='dataset2', values='corr'), annot=True, cmap='crest', fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f'../results/{args.task}_archwise_pr_rank.pdf', facecolor='w', bbox_inches="tight")
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
    rank_csv_file                = '../results/archwise_ranks.csv'    
    rank_df.to_csv(rank_csv_file, index = False)

    aggregated_ranks             = mc4_aggregator(rank_csv_file, header_row = 0, index_col = 0)
    
    with open(f'../results/{args.task}_archwise_aggregated_ranks.json', 'w') as wf:
        json.dump(aggregated_ranks, wf, indent=4)


def validate_hyps():

    hypothesis_cnt_dict = ddict(lambda: ddict(int))
    
    sig_df      = pd.read_csv(f'../results/{args.task}_sig_test.csv')
    
    models      = ['bertbc-adapter', 'bertbc-lora', 'bertbc-plain', 'bertlc-adapter', 'bertlc-lora', 'bertlc-plain', 'optb-adapter', 'optb-lora', 'optb-plain', 'optl-adapter' 'optl-lora', 'optl-plain',\
        'rb-adapter', 'rb-lora', 'rb-plain', 'rl-adapter', 'rl-lora', 'rl-plain', 't5b-adapter', 't5b-lora', 't5b-plain', 't5l-adapter', 't5l-lora', 't5l-plain']
    
    base_models = ['bertbc-adapter', 'bertbc-lora', 'bertbc-plain', 'optb-adapter',  'optb-lora', 'optb-plain',  'rb-adapter', 'rb-lora', 'rb-plain',  't5b-adapter', 't5b-lora', 't5b-plain']
    
    large_models = ['bertlc-adapter', 'bertlc-lora', 'bertlc-plain', 'optl-adapter', 'optl-lora', 'optl-plain', 'rl-adapter', 'rl-lora', 'rl-plain', 't5l-adapter', 't5l-lora', 't5l-plain']
    
    plain_models = ['bertbc-plain', 'bertlc-plain',  'optb-plain', 'optl-plain', 'rb-plain', 'rl-plain', 't5b-plain', 't5l-plain']
    
    lora_models  = ['bertbc-lora', 'bertlc-lora', 'optb-lora', 'optl-lora', 'rb-lora', 'rl-lora', 't5b-lora', 't5l-lora']
    
    adapter_models = ['bertbc-adapter', 'bertlc-adapter', 'optb-adapter', 'optl-adapter', 'rb-adapter', 'rl-adapter', 't5b-adapter', 't5l-adapter']
    
    eo_models      = ['bertbc-adapter', 'bertbc-lora', 'bertbc-plain', 'bertlc-adapter', 'bertlc-lora', 'bertlc-plain', 'rb-adapter', 'rb-lora', 'rb-plain', 'rl-adapter', 'rl-lora', 'rl-plain'] 
    
    ed_models      = ['t5b-adapter', 't5b-lora', 't5b-plain', 't5l-adapter', 't5l-lora', 't5l-plain']
    
    do_models      = ['optb-lora', 'optb-plain', 'optl-lora', 'optl-plain', 'optb-adapter', 'optl-adapter']
    
    analysis_cnt_dict = ddict(lambda: ddict(int))
    
    for index, row in sig_df.iterrows():
        dataset             = row['dataset']
        p_val               = row['p_val']
        better_model        = row['better_model']
        worse_model         = row['worse_model']
        
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
    
        # Plain vs LORA
    
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

        # Plain vs Adapter
        
        for pm, am in zip(plain_models, adapter_models):
            if am == better_model and pm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter']['better'] += 1
                    print(am, pm, p_val, 'better')
                else:
                    hypothesis_cnt_dict['adapter']['equal'] += 1
            
            elif am == worse_model and pm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter']['worse'] += 1
                else:
                    hypothesis_cnt_dict['adapter']['equal'] += 1
        
        # Adapter vs LORA
        
        
        
        for am, lm in zip(adapter_models, lora_models):
            if am == better_model and lm == worse_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter vs lora']['better'] +=1
                else:
                    hypothesis_cnt_dict['adapter vs lora']['equal'] +=1
        
            if am == worse_model and lm == better_model:
                if p_val < 0.05:
                    hypothesis_cnt_dict['adapter vs lora']['worse'] +=1
                else:
                    hypothesis_cnt_dict['adapter vs lora']['equal'] +=1
        
    
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
        'newsqa': 'OOD',
        'adversarial_squad': 'ROB',
        'adversarial_hotpotqa': 'ROB',
        'musique': 'COMP',
        'squad': 'A'
    }
    
    
    with open(f'../results/instancewise_{args.task}_dict.json', 'r') as f:
        acc_dict = json.load(f)
        
    mrc_datasets    = ['newsqa', 'adversarial_squad', 'adversarial_hotpotqa', 'musique', 'squad']
    
    results_dict     = ddict(list)
            
    for i, dataset in enumerate(mrc_datasets):
        for model in acc_dict[dataset]:
            
            tgt_acc = acc_dict[dataset][model]
            src_acc = acc_dict['squad'][model]
            
            results_dict['tgt_acc'].append(tgt_acc)
            results_dict['src_acc'].append(src_acc)
            results_dict['Gen_type'].append(gen_type_dict[dataset])
            results_dict['NSD'].append((tgt_acc - src_acc) / src_acc)
            
            model_cat = get_model_cat(model)
            results_dict['arch'].append(model_cat['arch'])
            results_dict['ft'].append(model_cat['ft'])
            results_dict['size'].append(model_cat['size'])
    
    results_df      = pd.DataFrame(results_dict)
    results_df.to_csv(f'../results/{args.task}_acc_results.csv', index=False)

    
    sns.histplot(data=results_df, x='tgt_acc', kde=True, bins=50)
    plt.savefig('../results/tgt_acc_hist.pdf', facecolor='w', bbox_inches="tight")
    
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols 
    
    reg_model = ols('tgt_acc ~ C(Gen_type) + C(arch) + C(ft) + C(size)', data=results_df).fit()
    anova_table = sm.stats.anova_lm(reg_model, typ=2)
    print(anova_table)    
    print(reg_model.params)

    # model = ols('tgt_acc ~ C(Gen_type)*C(arch)*C(ft)*C(size)*src_acc', data=results_df).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)
    # print(model.params)    

    reg_model = ols('NSD ~ C(Gen_type) + C(arch) + C(ft) + C(size)', data=results_df).fit()
    anova_table = sm.stats.anova_lm(reg_model, typ=2)
    print(anova_table)    
    print(reg_model.params)
    

    # reg_model = ols('NSD ~ C(Gen_type)* C(arch) * C(ft) * C(size)', data=results_df).fit()
    # anova_table = sm.stats.anova_lm(reg_model, typ=2)
    # print(anova_table)    
    # print(reg_model.params)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute significance test')
    parser.add_argument('--task',       type=str, default='mrc', help='task')
    parser.add_argument('--inp_dir',    type=str, default='json', help='input directory of predictions')
    parser.add_argument('--step',       type=str, default='analyse', required=True)
    
    args        =  parser.parse_args()
    
    if args.step == 'create':
        create_mrc_dict()
    elif args.step == 'sigtest':
       sigtest()
    elif args.step == 'inst':
        create_instancewise_dict()
        create_instancewise_corr()       
        
    elif args.step == 'arch':
        create_architecturewise_correlations()
    
    elif args.step == 'val_hyps':
        validate_hyps()
    
    elif args.step == 'change_stance':
        compute_sig_change_stances()
    
    elif args.step == 'anova':
        comp_anova()