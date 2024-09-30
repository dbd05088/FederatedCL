from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np
from collections import defaultdict
import hashlib

np.random.seed(42)

dir = 'dataset/DiDeMoSV'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)

meta_data = full_data['metadata']
full_data = full_data['data']

train_test_ratio = 0.2

# Function to compute a hash for each sample
def compute_sample_hash(item):
    context = item['task_instance']['context']
    response = item['response']
    hash_string = f"{context}{response}"
    return hashlib.md5(hash_string.encode()).hexdigest()

# Group samples by their content hash
sample_groups = defaultdict(list)
for idx, item in enumerate(full_data):
    sample_hash = compute_sample_hash(item)
    sample_groups[sample_hash].append(idx)

# Split groups into train and test
group_keys = list(sample_groups.keys())
np.random.shuffle(group_keys)
split_index = int(len(group_keys) * train_test_ratio)
test_groups = group_keys[:split_index]
train_groups = group_keys[split_index:]

test_idx = [idx for group in test_groups for idx in sample_groups[group]]
train_idx = [idx for group in train_groups for idx in sample_groups[group]]

# Create directories
for subset in ['train', 'test']:
    subset_folder = os.path.join(dir, subset)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

task_idx = 0
train_json_data = []
test_json_data = []

# Process and save data
for idx in range(len(full_data)):
    item = full_data[idx]
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    question = item['task_instance']['context']
    for i in range(len(new_item['image'])):
        rmv_i = '{image#%d}' % (i+1)
        rmv_t = '{table#%d}' % (i+1)
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

print(len(train_json_data))
print(len(test_json_data))

if len(train_json_data) > 10000:
    train_json_data = np.random.choice(train_json_data, size=10000, replace=False).tolist()
if len(test_json_data) > 2000:
    test_json_data = np.random.choice(test_json_data, size=2000, replace=False).tolist()

print(len(train_json_data))
print(len(test_json_data))

with open(f'{dir}/train/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(train_json_data, json_file, indent=4)
with open(f'{dir}/test/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(test_json_data, json_file, indent=4)
    
    
    
# from glob import glob
# import cv2

# os.makedirs('./dataset/DiDeMoSV/full/images_rgb', exist_ok=True)

# img_files = glob('./dataset/DiDeMoSV/full/images/*.jpg')
# for img_file in img_files:
#     file_name = img_file.split('/')[-1]
#     srcBGR = cv2.imread(img_file)
#     destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f'./dataset/DiDeMoSV/full/images_rgb/{file_name}', destRGB)

# # Renaming folders
# base_path = './dataset/DiDeMoSV/full'

# # Rename 'images' to 'images_bgr'
# os.rename(os.path.join(base_path, 'images'), os.path.join(base_path, 'images_bgr'))

# # Rename 'images_rgb' to 'images'
# os.rename(os.path.join(base_path, 'images_rgb'), os.path.join(base_path, 'images'))