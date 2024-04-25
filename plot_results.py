import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import json
warnings.filterwarnings(action='ignore')

plot_type = 'rouge_l' # loss, precision, bleu, meteor, rouge_l, cider
target_method_name = 'fedavg'
dir = f'./results/{target_method_name}/{target_method_name}/'
num_clients = 10


def print_from_log(filename):
    samples = []
    loss = []
    precision = []
    bleu = []
    meteor = []
    rouge_l = []
    cider = []
    # with open(exp_name+'/metrics.json', 'r') as fp:
    #     metrics = json.loads(fp)
    fp = open(filename+'.log', 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if "Test" in line:
            line_s = line.split(" ")
            samples.append(int(line_s[7]))
            loss.append(float(line_s[13]))
            precision.append(float(line_s[16]))
            bleu.append(float(line_s[19][1:-1]))
            meteor.append(float(line_s[30][1:-1]))
            rouge_l.append(float(line_s[33][1:-1]))
            cider.append(float(line_s[36][1:-1]))
    return {
        'samples': samples,
        'loss': loss,
        'precision':precision,
        'bleu':bleu,
        'meteor':meteor,
        'rouge_l':rouge_l,
        'cider':cider,
    }

for client_id in range(num_clients):
    if client_id in [7,8,9]:
        continue
    results_dict = print_from_log(dir + str(client_id) + '_client')

    
    plt.plot(results_dict['samples'], results_dict[plot_type], label=str(client_id), linewidth=2.0)
    # plt.plot(samples, precision, label=str(client_id), linewidth=2.0)
    
# server results
server_results = {}
datanames = ['AQUA', 'HRVQA-1.0', 'VQA-MED']

fp = open(dir + "server.log", 'r')
lines = fp.readlines()
fp.close()

for dataname in datanames:
    iter = 1
    if dataname == 'AQUA':
        samples = [0]
        loss = [6.3117]
        precision = [0.3831]
        bleu = [0.251941]
        meteor = [0.0621259]
        rouge_l = [0.141933]
        cider = [0.3564348]
    elif dataname == 'HRVQA-1.0':
        samples = [0]
        loss = [8.4556]
        precision = [0.3643]
        bleu = [0.0319448227]
        meteor = [0.034045510227]
        rouge_l = [0.035888372]
        cider = [0.087176975]
    elif dataname == 'VQA-MED':
        samples = [0]
        loss = [5.9314]
        precision = [0.4690]
        bleu = [0.124068479]
        meteor = [0.0335561]
        rouge_l = [0.076864]
        cider = [0.169718066]
    else:
        samples = []
        loss = []
        precision = []
        bleu = []
        meteor = []
        rouge_l = []
        cider = []
    for line in lines:
        if "Test" in line and dataname in line:
            line_s = line.split(" ")
            samples.append(int(800*iter))
            loss.append(float(line_s[7]))
            precision.append(float(line_s[10]))
            bleu.append(float(line_s[13][1:-1]))
            meteor.append(float(line_s[24][1:-1]))
            rouge_l.append(float(line_s[27][1:-1]))
            cider.append(float(line_s[30][1:-1]))
            iter+=1
    server_results[dataname] = {
        'samples': samples,
        'loss': loss,
        'precision':precision,
        'bleu':bleu,
        'meteor':meteor,
        'rouge_l':rouge_l,
        'cider':cider,
    }

for dataname in datanames:
    if dataname == 'VQA-MED':
        continue
    plt.plot(server_results[dataname]['samples'], server_results[dataname][plot_type], label=f'server-{dataname}', linewidth=2.0)

for i in range(0, 3200, 200):
    plt.axvline(x=i, color='b', linestyle='--', linewidth=0.1)

plt.axvline(x=0, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=800, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=1600, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=2400, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=3200, color='r', linestyle='--', linewidth=0.5)

plt.xlabel("# of samples", fontsize=20)
plt.ylabel(plot_type, fontsize=20)
plt.legend()
plt.title(f"Method results - {plot_type}", fontsize=20)
plt.savefig(f"{target_method_name}_results_plot - {plot_type}.png")
plt.clf()