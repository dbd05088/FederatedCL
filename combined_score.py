import jsonlines
import json

mode = 'fedavg'
Method = 'fedavg_1e-4_bs16_itr100_constant_round10_42'
# Method = 'fedavg_exscicap_1e-4_bs16_itr100_constant_round10_0'
# Method = 'sft_llava_sc12_lr1e-4_bs16_itr100_constant_round10'
# Method = 'fedper_8_llava_sc12_lr1e-4_bs16_itr100_constant_round10'
num_clients = 10
num_rounds = 5
is_client = True

datalist = ['Bongard-OpenWorld-0','Birds-to-Words-0', 'HRVQA-2', 'DiDeMoSV-0', 'NLVR2-0', 'Mementos-0', 'SciCap-1','AQUA-0', 'HRVQA-3', 'AQUA-1']
precision_datalist = {
    'HRVQA-2':2,
    'NLVR2-0':4,
    'AQUA-0':7,
    'HRVQA-3':8,
}
gpt_datalist = {
    'Bongard-OpenWorld-0':0,
    'Birds-to-Words-0':1,
    'DiDeMoSV-0':3,
    'Mementos-0':5,
    'SciCap-1':6,
    'AQUA-1':9}

scores = {name:0 for name in datalist}
if is_client:
    
    for data_name, id in precision_datalist.items():
        with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_rounds}_{data_name}.json', 'r') as fp:
            result = json.load(fp)[-1]
        scores[data_name] = result['precision']

    for data_name, id in gpt_datalist.items():
        count = 0
        total_score = 0
        with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/client{id}_round{num_rounds}_{data_name}.jsonl') as read_file:
            for line in read_file.iter():
                total_score += line['score']
                count+=1
            scores[data_name] = total_score/count/10
else:
    for data_name, id in precision_datalist.items():
        with open(f'./eval_results/{mode}/{Method}/server_round{num_rounds}_{data_name}.json', 'r') as fp:
            result = json.load(fp)[-1]
        scores[data_name] = result['precision']

    for data_name, id in gpt_datalist.items():
        count = 0
        total_score = 0
        with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}_correctness/server_round{num_rounds}_{data_name}.jsonl') as read_file:
            for line in read_file.iter():
                total_score += line['score']
                count+=1
            scores[data_name] = total_score/count/10


print(f"Method: {Method}")
print(f"Score: {scores}")
print(f"Avg Score: {sum(score for score in scores.values())/len(scores)}")