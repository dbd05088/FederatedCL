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

TOKEN = '<image>'

dir = 'dataset/mPLUG'
tasks = sorted(glob.glob(dir + '/tasks/*/*'))

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

invalid_file = open('./preprocess/mPLUG_invalid.txt', 'r')
invalid_img_files = invalid_file.readlines()
invalid_file.close()

# with open('./invalid_mPLUG_data.json', 'r') as fp:
#     invalid_json = json.load(fp)

for i in range(len(invalid_img_files)):
    if i < len(invalid_img_files)-1:
        invalid_img_files[i] = invalid_img_files[i][2:-1] 
    else:
        invalid_img_files[i] = invalid_img_files[i][2:] 
print(f'total invalid image file num: {len(invalid_img_files)}')
print(tasks)
task_idx = 0
total_removed_num = 0
for task in tasks:
    with open(task+'/train.json', 'r') as fp:
        train_json_data = json.load(fp)
    with open(task+'/test.json', 'r') as fp:
        test_json_data = json.load(fp)

    for jsondata in [train_json_data, test_json_data]:
        idx_to_del = []
        print(len(jsondata))
        for idx in range(len(jsondata)):
            if len(jsondata[idx]['image']) == 0:
                idx_to_del.append(idx)
            else:
                # check image token invalidity
                valid_img_num = len(jsondata[idx]['image'])
                input_text = jsondata[idx]['conversations'][0]['value']
                count = input_text.count(TOKEN)
                if count != valid_img_num:
                    idx_to_del.append(idx)
                    continue
                # check for img invalidity
                for img in jsondata[idx]['image']:
                    if img in invalid_img_files:
                        # print(img)
                        idx_to_del.append(idx)
                        break
                
                
        for idx in reversed(idx_to_del):
            del jsondata[idx]
        print(len(jsondata))
        total_removed_num += len(idx_to_del)
        
    print(len(train_json_data))
    print(len(test_json_data))

    if len(train_json_data) > 10000:
        train_json_data = np.random.choice(train_json_data, size=10000, replace=False).tolist()
    if len(test_json_data) > 2000:
        test_json_data = np.random.choice(test_json_data, size=2000, replace=False).tolist()

    print("final",len(train_json_data))
    print("final",len(test_json_data))
    

    with open(f'./dataset/mPLUG/train/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(train_json_data, json_file, indent=4)
    with open(f'./dataset/mPLUG/test/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(test_json_data, json_file, indent=4)
    task_idx += 1
print(total_removed_num)