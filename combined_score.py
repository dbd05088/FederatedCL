import jsonlines
import json
import csv
from collections import defaultdict
import os

mode = 'apfl'
# Method = 'llava_zeroshot_full'
# Method = 'PT_fullmem_sc20_lr1e-2_4tasks_5rounds_itr125'
# Method = 'fedavg_exscicap_1e-4_bs16_itr100_constant_round10_0'
# Method = 'sft_llava_sc12_lr1e-4_bs16_itr100_constant_round10'
# Method = 'fedper_8_llava_sc12_lr1e-4_bs16_itr100_constant_round10'
Method = 'apfl_sc12_lr1e-4_1e-6_itr100_round10'
num_rounds = 10
is_client = True

scenario_num = 20
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

scores = {}
client_scores = defaultdict(list)

for client_data in scenario:
    id = client_data['client_id']
    for data in client_data['datasets']:
        data_name = f"{data['dataset']}-{data['subset_id']}"
        
        if data['type'] == 'open-ended':
            count = 0
            total_score = 0
            if is_client:
                with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/client{id}_round{num_rounds}_{data_name}.jsonl') as read_file:
                    for line in read_file.iter():
                        total_score += line['score']
                        count+=1
            else:
                with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/server_round{num_rounds}_{data_name}.jsonl') as read_file:
                    for line in read_file.iter():
                        total_score += line['score']
                        count+=1
            score = total_score/count/10
        elif data['type'] == 'multi-choice':
            if is_client:
                with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_rounds}_{data_name}.json', 'r') as fp:
                    result = json.load(fp)[-1]
            else:
                with open(f'./eval_results/{mode}/{Method}/server_round{num_rounds}_{data_name}.json', 'r') as fp:
                    result = json.load(fp)[-1]
            score = result['accuracy']
        
        scores[data_name] = score
        client_scores[id].append(score)

avg_score = sum(scores.values()) / len(scores)
client_avg_scores = {id: sum(scores) / len(scores) for id, scores in client_scores.items()}

# Prepare data for CSV
csv_data = [Method, avg_score]
csv_data.extend([client_avg_scores.get(i, '') for i in range(10)])  # Assuming client IDs are 1-10
csv_data.extend(scores.values())

# Prepare header
header = ['method', 'final score']
header.extend([f'client {i}' for i in range(10)])
header.extend(scores.keys())

# Write to CSV
csv_file = 'results.csv'
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(csv_data)

print(f"Method: {Method}")
print(f"Score: {scores}")
print(f"Avg Score: {avg_score}")
print(f"Results have been appended to {csv_file}")

# datalist = ['Bongard-OpenWorld-0','Birds-to-Words-0', 'HRVQA-2', 'DiDeMoSV-0', 'NLVR2-0', 'Mementos-0', 'SciCap-1','AQUA-0', 'HRVQA-3', 'AQUA-1']
# precision_datalist = {
#     'HRVQA-2':2,
#     'NLVR2-0':4,
#     'AQUA-0':7,
#     'HRVQA-3':8,
# }
# gpt_datalist = {
#     'Bongard-OpenWorld-0':0,
#     'Birds-to-Words-0':1,
#     'DiDeMoSV-0':3,
#     'Mementos-0':5,
#     'SciCap-1':6,
#     'AQUA-1':9}

# scores = {name:0 for name in datalist}
# if is_client:
    
#     for data_name, id in precision_datalist.items():
#         with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_rounds}_{data_name}.json', 'r') as fp:
#             result = json.load(fp)[-1]
#         scores[data_name] = result['precision']

#     for data_name, id in gpt_datalist.items():
#         count = 0
#         total_score = 0
#         with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/client{id}_round{num_rounds}_{data_name}.jsonl') as read_file:
#             for line in read_file.iter():
#                 total_score += line['score']
#                 count+=1
#             scores[data_name] = total_score/count/10
# else:
#     for data_name, id in precision_datalist.items():
#         with open(f'./eval_results/{mode}/{Method}/server_round{num_rounds}_{data_name}.json', 'r') as fp:
#             result = json.load(fp)[-1]
#         scores[data_name] = result['precision']

#     for data_name, id in gpt_datalist.items():
#         count = 0
#         total_score = 0
#         with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/server_round{num_rounds}_{data_name}.jsonl') as read_file:
#             for line in read_file.iter():
#                 total_score += line['score']
#                 count+=1
#             scores[data_name] = total_score/count/10
# avg_score = sum(scores.values()) / len(scores)
# print(f"Method: {Method}")
# print(f"Score: {scores}")
# print(f"Avg Score: {avg_score}")

