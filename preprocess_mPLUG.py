# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil

dir = 'dataset/mPLUG'
tasks = sorted(glob.glob(dir + '/tasks/*/*'))

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

print(tasks)
idx = 0
for task in tasks:
    with open(task+'/train.json', 'r') as fp:
        train_json_data = json.load(fp)
    with open(task+'/test.json', 'r') as fp:
        test_json_data = json.load(fp)
    for item in train_json_data:
        if len(item['image']) == 0:
            del item['image']
    for item in test_json_data:
        if len(item['image']) == 0:
            del item['image']
    with open(f'./dataset/mPLUG/train/dataset-{idx}.json', 'w') as json_file:
        json.dump(train_json_data, json_file, indent=4)
    with open(f'./dataset/mPLUG/test/dataset-{idx}.json', 'w') as json_file:
        json.dump(test_json_data, json_file, indent=4)
    idx += 1
    
