# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import random
import numpy as np
import itertools
import jsonlines
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F

random.seed(123)
np.random.seed(123)

num_per_set = 3
num_combination_per_sample = 14
prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where "positive" images can be summarized as 1 "common" sentence and "negative" images cannot, the "common" sentence describes a set of concepts that are common to "positive" images. Please give the "common" sentence from "positive" images'''

def save_dataset(dataset_name, output_folder, subset_name):
    if subset_name == 'train':
        sample_size = 10000
    elif subset_name == 'test':
        sample_size = 2000
        
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
        
    with open(f"{output_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    json_data_list = []
    
    for item in datalist:
        
        answer = item['caption']
        positive_imgfiles = item['imageFiles'][:7]
        negative_imgfiles = item['imageFiles'][7:]
        positive_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in positive_imgfiles]
        negative_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in negative_imgfiles]
        
        arr = np.arange(len(positive_imgfiles))
        nCr = list(itertools.combinations(arr, num_per_set))
        random.shuffle(nCr)
        
        for idx, index in enumerate(nCr[:num_combination_per_sample]):
            index = np.array(list(index))
            random.shuffle(index)
            imgs = [positive_imgfiles[i] for i in index] + [negative_imgfiles[i] for i in index]
        
            answer = item['caption']
        
            # Structure for LLaVA JSON
            json_data = {
                "id": item['uid'] + "-" + str(idx),
                "image": imgs,#" |sep| ".join(imgs),
                "conversations": [
                    {
                        "from": "human",
                        "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\n" + prompts
                        # "value": "<image>"*len(imgs) + "\n" + prompts
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            json_data_list.append(json_data)

    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-0.json')
    print(len(json_data_list))
    if len(json_data_list) > sample_size:
        json_data_list = np.random.choice(json_data_list, size=sample_size, replace=False).tolist()
        print(len(json_data_list))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset/Bongard-OpenWorld'

# preprocess jsonl to json
# train (combine train and val)
train_data = []
with jsonlines.open(f"{output_folder}/train.jsonl") as f:
    for line in f.iter():
        train_data.append(line)

with jsonlines.open(f"{output_folder}/val.jsonl") as f:
    for line in f.iter():
        train_data.append(line)

with open(f"{output_folder}/train.json", 'w') as json_file:
    json.dump(train_data, json_file, indent=4)  



# test
test_data = []
with jsonlines.open(f"{output_folder}/test.jsonl") as f:
    for line in f.iter():
        test_data.append(line)

with open(f"{output_folder}/test.json", 'w') as json_file:
    json.dump(test_data, json_file, indent=4)  

save_dataset('Bongard-OpenWorld', output_folder, 'test')
save_dataset('Bongard-OpenWorld', output_folder, 'train')

