import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import copy
import csv
warnings.filterwarnings(action='ignore')

plot_types = ['recall', 'precision', 'bleu', 'meteor', 'rouge_l', 'cider']
Method = 'fedavg_bunny3b'
num_clients = 10
num_rounds = 10
do_plot = True

target_dirs = [
    './eval_results/fedavg/fedavg_eval/round_1.log',
    './eval_results/fedavg/fedavg_eval_round2/round_2.log',
    './eval_results/fedavg/fedavg_eval_round3/round_3.log',
    './eval_results/fedavg/fedavg_eval_round4/round_4.log',
    './eval_results/fedavg/fedavg_eval_round5/round_5.log',
    './eval_results/fedavg/fedavg_eval_round6/round_6.log',
    './eval_results/fedavg/fedavg_eval_round7/round_7.log',
    './eval_results/fedavg/fedavg_eval_round8/round_8.log',
    './eval_results/fedavg/fedavg_eval_round9/round_9.log',
    './eval_results/fedavg/fedavg_eval_round10/round_10.log',
]

datalist = ['HRVQA-0','HRVQA-1','HRVQA-2','HRVQA-3','HRVQA-4','HRVQA-5','HRVQA-6','HRVQA-7','HRVQA-8','HRVQA-9',]


single_dict = {'rounds':[],'precision':[], 'recall':[],
            'bleu':[],'meteor':[],'rouge_l':[],'cider':[]}
clients = {id:copy.deepcopy(single_dict) for id in range(num_clients)}
server = {dataname:copy.deepcopy(single_dict) for dataname in datalist}

for round, filename in enumerate(target_dirs):
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if "Test" in line:
            line_s = line.split(" ")
            if 'Client' in line:
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
    
    if do_plot:
        # client
        plt.clf()
        for client_id in range(num_clients):
            plt.plot(clients[client_id]['rounds'], clients[client_id][plot_type], label=str(client_id), linewidth=2.0)

        
        for i in range(num_rounds+1):
            plt.axvline(x=i, color='r', linestyle='--', linewidth=0.5)
        plt.xticks(list(range(num_rounds+1)))
        plt.xlabel("# of rounds", fontsize=20)
        plt.ylabel(plot_type, fontsize=20)
        plt.legend()
        plt.title(f"{Method} clients - {plot_type}", fontsize=20)
        plt.savefig(f"{Method}_clients_{plot_type}.png")
    
        # server
        plt.clf()
        for dataname, server_info in server.items():
            plt.plot(server_info['rounds'], server_info[plot_type], label=dataname, linewidth=2.0)
        
        for i in range(num_rounds+1):
            plt.axvline(x=i, color='r', linestyle='--', linewidth=0.5)
        plt.xticks(list(range(num_rounds+1)))
        plt.xlabel("# of rounds", fontsize=20)
        plt.ylabel(plot_type, fontsize=20)
        plt.legend()
        plt.title(f"{Method} server - {plot_type}", fontsize=20)
        plt.savefig(f"{Method}_server_{plot_type}.png")
    