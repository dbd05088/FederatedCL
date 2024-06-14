import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import copy
import csv
warnings.filterwarnings(action='ignore')

plot_types = ['recall', 'precision', 'bleu', 'meteor', 'rouge_l', 'cider'] # 
mode = 'fedavg'
Method = 'fedavg_llava_sc10_lr5e-5_bs16_nodist_itr50'
num_clients = 9
num_rounds = 10
do_plot = True
do_csv = False

target_dirs = [
    f'./eval_results/{mode}/{Method}/round_1.log',
    f'./eval_results/{mode}/{Method}/round_2.log',
    f'./eval_results/{mode}/{Method}/round_3.log',
    f'./eval_results/{mode}/{Method}/round_4.log',
    f'./eval_results/{mode}/{Method}/round_5.log',
    f'./eval_results/{mode}/{Method}/round_6.log',
    f'./eval_results/{mode}/{Method}/round_7.log',
    f'./eval_results/{mode}/{Method}/round_8.log',
    f'./eval_results/{mode}/{Method}/round_9.log',
    f'./eval_results/{mode}/{Method}/round_10.log',
]

# datalist = ['HRVQA-0','HRVQA-1','HRVQA-2','HRVQA-3','HRVQA-4','HRVQA-5','HRVQA-6','HRVQA-7','HRVQA-8','HRVQA-9',]
# datalist = ['Bongard-OpenWorld-0', 'Birds-to-Words-0', 'Describe-Diff-0', 'DiDeMoSV-0', 'NLVR2-0', 'HRVQA-7', 'Mementos-0', 'SciCap-0', 'SciCap-1', 'HRVQA-3']
# datalist = ['mPLUG-1','mPLUG-2','mPLUG-3','mPLUG-4','mPLUG-6','mPLUG-7',]
# datalist = ['Mementos-0', 'HRVQA-0', 'Describe-Diff-0', 'DiDeMoSV-0', 'NLVR2-0', 'HRVQA-7', 'Mementos-0', 'SciCap-0', 'SciCap-1', 'HRVQA-3']
# datalist = ['Bongard-OpenWorld-0','Birds-to-Words-0','DiDeMoSV-0','mPLUG-2', 'Mementos-0', 'AQUA-0']
datalist = ['Bongard-OpenWorld-0','Birds-to-Words-0', 'HRVQA-2', 'DiDeMoSV-0', 'NLVR2-0', 'Mementos-0', 'mPLUG-2','AQUA-0', 'HRVQA-3']

# single_dict = {'rounds':[],'precision':[], 'recall':[],
#             'bleu':[],'meteor':[],'rouge_l':[],'cider':[]}
# clients = {id:copy.deepcopy(single_dict) for id in range(num_clients)}
# server = {dataname:copy.deepcopy(single_dict) for dataname in datalist}

#llava_sc10_zeroshot
clients = {
    0: {'rounds':[0],'precision':[0.3205], 'recall':[0.1819],
            'bleu':[0.1369],'meteor':[0.10309],'rouge_l':[0.187913],'cider':[0.303591]},
    1: {'rounds':[0],'precision':[0.2604], 'recall':[0.1233],
            'bleu':[0.104565],'meteor':[0.0740335],'rouge_l':[0.11595],'cider':[0.003356]},
    2: {'rounds':[0],'precision':[0.3580], 'recall':[0.3908],
            'bleu':[0.081217],'meteor':[0.08475],'rouge_l':[0.08475],'cider':[0.211875]},
    3: {'rounds':[0],'precision':[0.4253], 'recall':[0.0801],
            'bleu':[0.040888],'meteor':[0.0704189],'rouge_l':[0.080415709],'cider':[0.00834768]},
    4: {'rounds':[0],'precision':[0.7571], 'recall':[0.7570],
            'bleu':[0.513885],'meteor':[0.513885],'rouge_l':[0.513885],'cider':[1.28471]},
    5: {'rounds':[0],'precision':[0.2793], 'recall':[0.3563],
            'bleu':[0.3046],'meteor':[0.1227],'rouge_l':[0.219165],'cider':[0.0453536]},
    6: {'rounds':[0],'precision':[0.1945], 'recall':[0.1925],
            'bleu':[0.1255157],'meteor':[0.058987],'rouge_l':[0.1091127],'cider':[0.038724]},
    7: {'rounds':[0],'precision':[0.2013], 'recall':[0.2595],
            'bleu':[0.08241],'meteor':[0.11799],'rouge_l':[0.13661],'cider':[0.313495]},
    8: {'rounds':[0],'precision':[0.8937], 'recall':[0.8985],
            'bleu':[0.679749],'meteor':[0.67975],'rouge_l':[0.67975],'cider':[1.699375]},
}

server = {
    datalist[0]: {'rounds':[0],'precision':[0.3205], 'recall':[0.1819],
            'bleu':[0.1369],'meteor':[0.10309],'rouge_l':[0.187913],'cider':[0.303591]},
    datalist[1]: {'rounds':[0],'precision':[0.2604], 'recall':[0.1233],
            'bleu':[0.104565],'meteor':[0.0740335],'rouge_l':[0.11595],'cider':[0.003356]},
    datalist[2]: {'rounds':[0],'precision':[0.3580], 'recall':[0.3908],
            'bleu':[0.081217],'meteor':[0.08475],'rouge_l':[0.08475],'cider':[0.211875]},
    datalist[3]: {'rounds':[0],'precision':[0.4253], 'recall':[0.0801],
            'bleu':[0.040888],'meteor':[0.0704189],'rouge_l':[0.080415709],'cider':[0.00834768]},
    datalist[4]: {'rounds':[0],'precision':[0.7571], 'recall':[0.7570],
            'bleu':[0.513885],'meteor':[0.513885],'rouge_l':[0.513885],'cider':[1.28471]},
    datalist[5]: {'rounds':[0],'precision':[0.2793], 'recall':[0.3563],
            'bleu':[0.3046],'meteor':[0.1227],'rouge_l':[0.219165],'cider':[0.0453536]},
    datalist[6]: {'rounds':[0],'precision':[0.1945], 'recall':[0.1925],
            'bleu':[0.1255157],'meteor':[0.058987],'rouge_l':[0.1091127],'cider':[0.038724]},
    datalist[7]: {'rounds':[0],'precision':[0.2013], 'recall':[0.2595],
            'bleu':[0.08241],'meteor':[0.11799],'rouge_l':[0.13661],'cider':[0.313495]},
    datalist[8]: {'rounds':[0],'precision':[0.8937], 'recall':[0.8985],
            'bleu':[0.679749],'meteor':[0.67975],'rouge_l':[0.67975],'cider':[1.699375]},
}

# clients = {
#     0: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0],'meteor':[0.0],'rouge_l':[0.0],'cider':[0.0]},
#     1: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0525],'meteor':[0.042],'rouge_l':[0.1262],'cider':[0.02486]},
#     2: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0816],'meteor':[0.05077],'rouge_l':[0.1290],'cider':[0.043]},
#     3: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.10416],'meteor':[0.0625],'rouge_l':[0.1289],'cider':[0.1605]},
#     4: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5929],'meteor':[0.59929],'rouge_l':[0.5999],'cider':[1.4962]},
#     5: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.09254],'meteor':[0.089776],'rouge_l':[0.09355],'cider':[0.222826]},
#     6: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2225],'meteor':[0.1028],'rouge_l':[0.2082],'cider':[0.0299]},
#     7: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0855],'meteor':[0.0372],'rouge_l':[0.0906],'cider':[0.0432]},
#     8: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.12088],'meteor':[0.0494],'rouge_l':[0.1083],'cider':[0.10895]},
#     9: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.6715],'meteor':[0.6715],'rouge_l':[0.6715],'cider':[1.67875]},
# }

# server = {
#     datalist[0]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0],'meteor':[0.0],'rouge_l':[0.0],'cider':[0.0]},
#     datalist[1]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0525],'meteor':[0.042],'rouge_l':[0.1262],'cider':[0.02486]},
#     datalist[2]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0816],'meteor':[0.05077],'rouge_l':[0.1290],'cider':[0.043]},
#     datalist[3]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.10416],'meteor':[0.0625],'rouge_l':[0.1289],'cider':[0.1605]},
#     datalist[4]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5929],'meteor':[0.59929],'rouge_l':[0.5999],'cider':[1.4962]},
#     datalist[5]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.09254],'meteor':[0.089776],'rouge_l':[0.09355],'cider':[0.222826]},
#     datalist[6]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2225],'meteor':[0.1028],'rouge_l':[0.2082],'cider':[0.0299]},
#     datalist[7]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0855],'meteor':[0.0372],'rouge_l':[0.0906],'cider':[0.0432]},
#     datalist[8]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.12088],'meteor':[0.0494],'rouge_l':[0.1083],'cider':[0.10895]},
#     datalist[9]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.6715],'meteor':[0.6715],'rouge_l':[0.6715],'cider':[1.67875]},
# }

# llava zeroshot
# clients = {
#     0: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2697],'meteor':[0.10618],'rouge_l':[0.18626],'cider':[0.0416]},
#     1: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5],'meteor':[0.5],'rouge_l':[0.5],'cider':[0.5]},
#     2: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0929],'meteor':[0.07793],'rouge_l':[0.11455],'cider':[0.00513]},
#     3: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.039446],'meteor':[0.06556],'rouge_l':[0.0816366],'cider':[0.0366159]},
#     4: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.46816],'meteor':[0.477022],'rouge_l':[0.476965],'cider':[1.191157]},
#     5: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0679387],'meteor':[0.0798],'rouge_l':[0.0797],'cider':[0.1952]},
#     6: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2697],'meteor':[0.10618],'rouge_l':[0.18626],'cider':[0.0416]},
#     7: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.12247],'meteor':[0.042159],'rouge_l':[0.08920],'cider':[0.02037]},
#     8: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.09613],'meteor':[0.04905],'rouge_l':[0.09268],'cider':[0.03857]},
#     9: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5579],'meteor':[0.56125],'rouge_l':[0.56125],'cider':[1.403125]},
# }

# server = {
#     datalist[0]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2697],'meteor':[0.10618],'rouge_l':[0.18626],'cider':[0.0416]},
#     datalist[1]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5],'meteor':[0.5],'rouge_l':[0.5],'cider':[0.5]},
#     datalist[2]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0929],'meteor':[0.07793],'rouge_l':[0.11455],'cider':[0.00513]},
#     datalist[3]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.039446],'meteor':[0.06556],'rouge_l':[0.0816366],'cider':[0.0366159]},
#     datalist[4]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.46816],'meteor':[0.477022],'rouge_l':[0.476965],'cider':[1.191157]},
#     datalist[5]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.0679387],'meteor':[0.0798],'rouge_l':[0.0797],'cider':[0.1952]},
#     datalist[6]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.2697],'meteor':[0.10618],'rouge_l':[0.18626],'cider':[0.0416]},
#     datalist[7]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.12247],'meteor':[0.042159],'rouge_l':[0.08920],'cider':[0.02037]},
#     datalist[8]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.09613],'meteor':[0.04905],'rouge_l':[0.09268],'cider':[0.03857]},
#     datalist[9]: {'rounds':[0],'precision':[], 'recall':[],
#             'bleu':[0.5579],'meteor':[0.56125],'rouge_l':[0.56125],'cider':[1.403125]},
# }

for round, filename in enumerate(target_dirs):
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if "Test" in line:
            line_s = line.split(" ")
            if 'Client' in line:# and len(clients[int(line_s[6][:1])]['rounds']) < num_rounds:
                id = int(line_s[6][:1])
                clients[id]['rounds'].append(round+1)
                clients[id]['precision'].append(float(line_s[12]))
                clients[id]['recall'].append(float(line_s[15]))
                clients[id]['bleu'].append(float(line_s[18][1:-1]))
                clients[id]['meteor'].append(float(line_s[29][1:-1]))
                clients[id]['rouge_l'].append(float(line_s[32][1:-1]))
                clients[id]['cider'].append(float(line_s[35][1:-1]))
            elif 'Server' in line:
                dataname = line_s[7]
                server[dataname]['rounds'].append(round+1)
                server[dataname]['precision'].append(float(line_s[10]))
                server[dataname]['recall'].append(float(line_s[13]))
                server[dataname]['bleu'].append(float(line_s[16][1:-1]))
                server[dataname]['meteor'].append(float(line_s[27][1:-1]))
                server[dataname]['rouge_l'].append(float(line_s[30][1:-1]))
                server[dataname]['cider'].append(float(line_s[33][1:-1]))
                

# csv file
fields = ['name',
    'mean_recall', 'final_recall',
    'mean_precision', 'final_precision',
    'mean_bleu', 'final_bleu',
    'mean_meteor', 'final_meteor',
    'mean_rouge_l', 'final_rouge_l',
    'mean_cider', 'final_cider',
]
rows = [['client'],['server']] # one for cilent, one for server
for plot_type in plot_types:
    if do_csv:
        # clients
        mean_per_client = [sum(clients[client_id][plot_type])/len(clients[client_id][plot_type]) for client_id in range(num_clients)]
        final_per_client = [clients[client_id][plot_type][-1] for client_id in range(num_clients)]
        rows[0].append(sum(mean_per_client)/len(mean_per_client))
        rows[0].append(sum(final_per_client)/len(final_per_client))
        
        # servr
        mean_per_server = [sum(server_info[plot_type])/len(server_info[plot_type]) for server_info in server.values()]
        final_per_server = [server_info[plot_type][-1] for server_info in server.values()]
        rows[1].append(sum(mean_per_server)/len(mean_per_server))
        rows[1].append(sum(final_per_server)/len(final_per_server))
        
        # name of csv file
        filename = f"{Method}.csv"
        
        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
        
            # writing the fields
            csvwriter.writerow(fields)
        
            # writing the data rows
            csvwriter.writerows(rows)
# for plot_type in plot_types:
#     if do_plot:
#         # client
#         plt.clf()
#         fig = plt.figure(figsize=(6.5, 5.5))
#         for client_id in range(num_clients):
#             plt.plot(clients[client_id]['rounds'], clients[client_id][plot_type], label=datalist[client_id], linewidth=2.0)
#         for i in range(num_rounds+1):
#             plt.axvline(x=i, color='r', linestyle='--', linewidth=0.5)
#         plt.xticks(list(range(num_rounds+1)))
#         plt.xlabel("# of rounds", fontsize=20)
#         plt.ylabel(plot_type, fontsize=20)
#         plt.legend()
#         plt.title(f"{Method} clients - {plot_type}", fontsize=20)
#         plt.savefig(f"{Method}_clients_{plot_type}.png")
    
#         # server
#         plt.clf()
#         fig = plt.figure(figsize=(6.5, 5.5))
#         for dataname, server_info in server.items():
#             plt.plot(server_info['rounds'], server_info[plot_type], label=dataname, linewidth=2.0)
        
#         for i in range(num_rounds+1):
#             plt.axvline(x=i, color='r', linestyle='--', linewidth=0.5)
#         plt.xticks(list(range(num_rounds+1)))
#         plt.xlabel("# of rounds", fontsize=20)
#         plt.ylabel(plot_type, fontsize=20)
#         plt.legend()
#         plt.title(f"{Method} server - {plot_type}", fontsize=20)
#         plt.savefig(f"{Method}_server_{plot_type}.png")

if do_plot:
    num_rows = 2
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.suptitle(f"{Method}", fontsize=16)
    for i, plot_type in enumerate(plot_types):
        row = i // num_cols
        col = i % num_cols
        for client_id in range(num_clients):
            # if 'HRVQA' in datalist[client_id] or 'NLVR' in datalist[client_id]:
            #     continue
            axs[row, col].plot(clients[client_id]['rounds'], clients[client_id][plot_type], label=datalist[client_id], linewidth=2.0)

        for j in range(num_rounds + 1):
            axs[row, col].axvline(x=j, color='r', linestyle='--', linewidth=0.5)
        axs[row, col].set_xticks(list(range(num_rounds + 1)))
        axs[row, col].set_xlabel("# of rounds", fontsize=14)
        axs[row, col].set_ylabel(plot_type, fontsize=14)
        # axs[row, col].legend()
        axs[row, col].set_title(f"clients - {plot_type}", fontsize=14)

    # Create a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.97, 0.5), fontsize=12)


    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{Method}_clients_all_plots.png")
    
    #server
    plt.clf()
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.suptitle(f"{Method}", fontsize=16)
    for i, plot_type in enumerate(plot_types):
        row = i // num_cols
        col = i % num_cols
        for dataname, server_info in server.items():
            # if 'HRVQA' in dataname or 'NLVR' in dataname:
            #     continue
            # axs[row, col].plot(clients[client_id]['rounds'], clients[client_id][plot_type], label=datalist[client_id], linewidth=2.0)
            axs[row, col].plot(server_info['rounds'], server_info[plot_type], label=dataname, linewidth=2.0)

        for j in range(num_rounds + 1):
            axs[row, col].axvline(x=j, color='r', linestyle='--', linewidth=0.5)
        axs[row, col].set_xticks(list(range(num_rounds + 1)))
        axs[row, col].set_xlabel("# of rounds", fontsize=14)
        axs[row, col].set_ylabel(plot_type, fontsize=14)
        # axs[row, col].legend()
        axs[row, col].set_title(f"servers - {plot_type}", fontsize=14)

    # Create a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12)


    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{Method}_servers_all_plots.png")
        