# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np
np.random.seed(42)

dir = 'dataset/HRVQA'
tasks = sorted(glob.glob(dir + '/tasks/*'))

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

print(tasks)
idx = 0
global_options = {2:['A','B'], 3:['A','B','C'], 4:['A','B','C','D']}
    
for task in tasks:
    print(task)
    with open(task+'/train.json', 'r') as fp:
        train_json_data = json.load(fp)
    with open(task+'/test.json', 'r') as fp:
        test_json_data = json.load(fp)
    
    values = []
    for item in train_json_data:
        values.append(item['conversations'][1]['value'])
    for item in test_json_data:
        values.append(item['conversations'][1]['value'])
    
    unique_values = sorted(list(set(values)))
    option_num = min(4, len(unique_values))
    options = global_options[option_num]
    
    for jsondata in [train_json_data, test_json_data]:
        for item in jsondata:
            answer_idx = np.random.randint(0, len(options))
            answer = item['conversations'][1]['value']
            answer_idx_inlist = unique_values.index(answer)
            choices = list(np.random.choice(unique_values[:answer_idx_inlist] + unique_values[answer_idx_inlist+1:], size=len(options) - 1, replace=False))
            choices.insert(answer_idx, answer)
            
            for i in range(len(options)):
                item['conversations'][0]['value'] += "\n"
                item['conversations'][0]['value'] += options[i] + '. ' + choices[i]
            item['conversations'][1]['value'] = f"{options[answer_idx]}"
            item['conversations'][0]['value'] += '\n' + "Answer with the option's letter from the given choices directly."
    
    with open(f'./dataset/HRVQA/train/dataset-{idx}.json', 'w') as json_file:
        json.dump(train_json_data, json_file, indent=4)
    with open(f'./dataset/HRVQA/test/dataset-{idx}.json', 'w') as json_file:
        json.dump(test_json_data, json_file, indent=4)
    idx += 1