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

invalid_file = open('./mPLUG_invalid.txt', 'r')
invalid_img_files = invalid_file.readlines()
invalid_file.close()
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
                del jsondata[idx]['image']
            else:
                for img in jsondata[idx]['image']:
                    if img in invalid_img_files:
                        print(img)
                        idx_to_del.append(idx)
                        break
        for idx in reversed(idx_to_del):
            del jsondata[idx]
        print(len(jsondata))
        total_removed_num += len(idx_to_del)

    with open(f'./dataset/mPLUG/train/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(train_json_data, json_file, indent=4)
    with open(f'./dataset/mPLUG/test/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(test_json_data, json_file, indent=4)
    task_idx += 1
print(total_removed_num)