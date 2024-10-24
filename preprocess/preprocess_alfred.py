from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np

np.random.seed(42)

dir = 'dataset/ALFRED'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)

meta_data = full_data['metadata']
full_data = full_data['data']

total_len = len(full_data)

train_test_ratio = 0.2

idx_list = list(range(total_len))
test_idx = list(range(int(total_len*0.8), total_len))

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

json_data_list = []

for idx in range(total_len):
    item = full_data[idx]
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    
    if len(new_item['image']) > 7:
        breakpoint()
    
    question = item['task_instance']['context']
    for i in range(len(new_item['image'])):
        rmv_i = '{image#%d}'% (i+1)
        rmv_t = '{table#%d}'% (i+1)
        question = question.replace(rmv_i, '<image>')
        question = question.replace(rmv_t, '<image>')
    
    new_item['conversations'] = [
        {
            "from": "human",
            "value": meta_data['task_instruction'][item['task_instruction_id']] + question
        },
        {
            "from": "gpt",
            "value": item['response']
        }
    ]
    json_data_list.append(new_item)

json_data_train = json_data_list[:int(total_len*0.8)]
json_data_test = json_data_list[int(total_len*0.8):]

print(len(json_data_train))
print(len(json_data_test))

if len(json_data_train) > 10000:
    json_data_train = np.random.choice(json_data_train, size=10000, replace=False).tolist()
if len(json_data_test) > 2000:
    json_data_test = np.random.choice(json_data_test, size=2000, replace=False).tolist()

print(len(json_data_train))
print(len(json_data_test))

with open(f'{dir}/train/dataset-0.json', 'w') as json_file:
    json.dump(json_data_train, json_file, indent=4)
with open(f'{dir}/test/dataset-0.json', 'w') as json_file:
    json.dump(json_data_test, json_file, indent=4)