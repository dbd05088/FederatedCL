from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np

np.random.seed(42)

dir = 'dataset/DiDeMoSV'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)

meta_data = full_data['metadata']
full_data = full_data['data']

total_len = len(full_data)

train_test_ratio = 0.2

idx_list = list(range(total_len))
test_idx = np.random.choice(idx_list, size=int(total_len*0.2), replace=False).tolist()

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

task_idx = 0
train_json_data = []
test_json_data = []

for idx in range(total_len):
    item = full_data[idx]
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
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
    
    if idx in test_idx:
        test_json_data.append(new_item)
    else:
        train_json_data.append(new_item)

with open(f'{dir}/train/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(train_json_data, json_file, indent=4)
with open(f'{dir}/test/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(test_json_data, json_file, indent=4)
    
    
from glob import glob
import cv2

os.makedirs('./dataset/DiDeMoSV/full/images_rgb', exist_ok=True)

img_files = glob('./dataset/DiDeMoSV/full/images/*.jpg')
for img_file in img_files:
    file_name = img_file.split('/')[-1]
    srcBGR = cv2.imread(img_file)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'./dataset/DiDeMoSV/full/images_rgb/{file_name}', destRGB)

# Renaming folders
base_path = './dataset/DiDeMoSV/full'

# Rename 'images' to 'images_bgr'
os.rename(os.path.join(base_path, 'images'), os.path.join(base_path, 'images_bgr'))

# Rename 'images_rgb' to 'images'
os.rename(os.path.join(base_path, 'images_rgb'), os.path.join(base_path, 'images'))