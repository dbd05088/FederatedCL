import jsonlines
import json
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

mode_method_dict = {
    # 'Frozen weight':'sft_bs4_saveoptim_lr3e-3_sc8_1task_1round_fixitr1000',
    
    # 'FedOurs (round5)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round5_local',
    # 'FedOurs (round10)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round10_local',
    # 'FedOurs (round15)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round15_local',
    # 'FedOurs (round20)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round20_local',
    
    
    # 'FedAvg':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_fedavg_round5',
    # # 'FedOurs (Mean)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_fedours_mean_round15',
    # 'FedOurs (ExcludeMean)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_fedours_excludemean_round5',
    # 'FedOurs (gradsim,mid,t=0.2,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_16gradsimmean_t0.2_round5_local',
    # 'FedOurs (gradsim,last,t=0.2,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_lastgradsimmean_t0.2_round5_local',
    
    # 'FedOurs (grad,mid,t=0.1)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradmean_t0.1_round15',
    # 'FedOurs (grad,mid,t=0.2)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradmean_t0.2_round15',
    # 'FedOurs (grad,mid,t=0.5,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradmean_t0.5_round15_local',
    # 'FedOurs (gradsim,mid,t=0.1,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.1_round15_local',
    # 'FedOurs (gradsim,mid,t=0.1,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.1_round15_local',
    
    # 'FedOurs (gradsim,mid,t=0.2,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15_local',
    # 'FedOurs (gradsim,mid,t=0.5,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.5_round15_local',
    
    # 'FedOurs (fisher,last,t=0.2)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastfishermean_t0.2_round15',
    # 'FedOurs (fisher,last,t=0.2,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastfishermean_t0.2_round15_local',
    # 'FedOurs (fishersim,last,t=0.2)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastfishersimmean_t0.2_round15',
    
    # 'FedOurs (fishersim,last,t=0.2,local)':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastfishersimmean_t0.2_round15_local',
    
    
    # 'FedOurs (grad,last,t=0.1)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradmean_t0.1_round15',
    # 'FedOurs (grad,last,t=0.2,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradmean_t0.2_round15_local',
    # 'FedOurs (gradsim,last,t=0.1)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.1_round15',
    # 'FedOurs (gradsim,last,t=0.1,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.1_round15_local',
    # 'FedOurs (gradsim,last,t=0.2)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_round15',
    
    # 'FedOurs (gradsim,last,t=1.0,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t1.0_round15_local',
    # 'FedOurs (gradsim,last,t=0.5,local)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.5_round15_local',
    # ## sc8 best ##
    
    # 'SFT+gradsim': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sft_16gradsimmean_t0.2_round15',
    # 'FedAvg+gradsim':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_fedavg_16gradsimmean_t0.2_round15',
    
    # 'FedOurs (last)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_round15_local',
    # 'FedOurs (mid) bs4': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15_local',
    # 'FedOurs (mid) bs8': 'sft_bs8_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15',
    # 'FedOurs (mid) bs16': 'sft_bs16_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15',
    
    # 'Frozen weight':'sft_bs4_saveoptim_lr3e-3_sc8_1task_1round_fixitr1000',
    # 'FedOurs KL (mid, round15)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_kl_16gradsimmean_t0.2_round15_local',
    # 'FedOurs KL (mid, round20)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_kl_16gradsimmean_t0.2_round20_local',
    # 'FedOurs KL (last, round15)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_kl_lastgradsimmean_t0.2_round15_local',
    # 'FedOurs KL (last, round20)': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_kl_lastgradsimmean_t0.2_round20_local',
    # 'FedAvg':'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_2_fedavg_round25',
    # 'FedOurs': 'sft_bs4_saveoptim_lr3e-3_sc8_1tasks_1rounds_fixitr1000_sc40_2_16gradsimmean_t0.2_round25_local',
    
    # 'Frozen weight':'sft_bs4_saveoptim_lr3e-3_sc10_1task_1round_fixitr1000',
    # 'FedOurs KL (mid, round15)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_kl_16gradsimmean_t0.2_round15_local',
    # 'FedOurs KL (mid, round20)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_kl_16gradsimmean_t0.2_round20_local',
    # 'FedOurs KL (last, round15)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_kl_lastgradsimmean_t0.2_round15_local',
    # 'FedOurs KL (last, round20)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_kl_lastgradsimmean_t0.2_round20_local',
    # 'FedAvg':'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_sc40_2_fedavg_round25',
    # 'FedOurs': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_sc40_2_16gradsimmean_t0.2_round25_local',
    # # 'FedOurs (Mean)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_fedours_mean_round15',
    # 'FedOurs (ExcludeMean)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_sc40_fedours_excludemean_round5',
    # # 'FedOurs (grad,last,t=0.2)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradmean_t0.2_round15',
    # 'FedOurs (gradsim,mid,t=0.2,local)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_sc40_16gradsimmean_t0.2_round5_local',
    # 'FedOurs (round5)':'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round5_local',
    # 'FedOurs (round10)':'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round10_local',
    # 'FedOurs (round15)':'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round15_local',
    # 'FedOurs (round20)':'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.2_2_round20_local',
    # 'FedOurs (gradsim,last,t=1.0,local)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t1.0_round15_local',
    # 'FedOurs (gradsim,last,t=0.5,local)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_lastgradsimmean_t0.5_round15_local',
    
    # 'FedOurs (gradsim,last,t=0.2,local)': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_sc40_lastgradsimmean_t0.2_round5_local',
    
    'FedOurs (mid) bs4': 'sft_bs4_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15_local',
    'FedOurs (mid) bs8': 'sft_bs8_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15',
    'FedOurs (mid) bs16': 'sft_bs16_saveoptim_lr3e-3_sc10_1tasks_1rounds_fixitr1000_16gradsimmean_t0.2_round15',
}

# colors = ['#3A1730', '#C18A3D', '#588157', '#E63946', '#BCBD22', '#17BECF', '#457B9D']
colors = ['#457B9D', '#314832', '#D8CFC0', '#E63946', '#3A1730', '#C18A3D', '#588157', '#38322C', '#BCBD22', '#17BECF']

mode_color_dict = {
    'sft': colors[3],
    'fedavg': colors[1],
    'fedours': colors[0],
}

scenario_num = 10
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 995]
# iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350]
# iters = [0, ,5, 10, 20, 30, 40, 50, 100]
num_rounds = 1

# plot_mode = 'seen_task'
data_indices = [
    [0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],
]

for client_data in scenario:
    id = client_data['client_id']
    mode_scores = {}
    for mode in mode_method_dict.keys():
        Method = mode_method_dict[mode]
        client_scores = []
        for num_round in range(num_rounds):
            data_index = data_indices[num_round]
            done = False
            for iter in iters:
                summed_score = 0
                for d_idx in data_index:
                    data = client_data['datasets'][d_idx]
                    data_name = f"{data['dataset']}-{data['subset_id']}"
                    
                    try:
                        if iter == 0 and mode == 'Frozen weight':
                            filename = f'./eval_results/zeroshot/zeroshot_sc{scenario_num}/client{id}_round20_{data_name}.json'
                        else:
                            filename = f'./eval_results/{Method.split("_")[0]}/{Method}/client{id}_round{num_round+1}_iter{iter}_{data_name}.json'
                        with open(filename, 'r') as fp:
                            result = json.load(fp)[-1]
                    except Exception as e:
                        print(e)
                        done = True
                        break
                    
                    if data['type'] == 'multi-choice':
                        score = result['accuracy']
                    elif data['type'] == 'open-ended':
                        if data['metric'] == 'F1':
                            score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                        elif data['metric'] == 'RougeL':
                            score = result['ROUGE_L'][0]
                        elif data['metric'] == 'cider':
                            score = result['CIDEr'][0]
                    summed_score += score
                if done:
                    break
                client_scores.append(summed_score / len(data_index))                    
        mode_scores[mode] = client_scores
        
    # Plotting the scores
    plt.figure(figsize=(8, 4.8))
    plt.axes().set_facecolor("#F5F5F5")
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    y = iters#[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for mode, scores in mode_scores.items():
        plt.plot(y[:len(scores)], scores, label=f'{mode}', linewidth=2.0)#, color=mode_color_dict[mode])#, marker='o')
    
    plt.title(f'{data_name} Scores', fontsize=20)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.legend(fontsize=14)
    # plt.grid(axis='y')
    plt.grid(True)
    

    # Save the plot
    plt.savefig(f'plot_unseen_train_client_{id}_sc{scenario_num}.png')