# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
# import random
import jsonlines
import numpy as np

np.random.seed(42)
# random.seed(42)

num_per_set = 3
prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where "positive" images share "common" visual concepts and "negative" images cannot, the "common" visual concepts exclusively depicted by the "positive" images. And then given 1 "query" image, please determine whether it belongs to "positive" or "negative".'''

def save_dataset(dataset_name, output_folder, subset_name):
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
        
    with open(f"{output_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    json_data_list = []
    
    for item in datalist:
        # 6 positive imgs, 1 positive query, 6 negative imgs, 1 negative query
        # --> 2 pos, 2 neg
        # how many data from this single item?
        
        
        answer = item['caption']
        positive_imgfiles = item['imageFiles'][:7]
        negative_imgfiles = item['imageFiles'][7:]
        positive_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in positive_imgfiles]
        negative_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in negative_imgfiles]
        
        positive_files = positive_imgfiles[:-1]
        positive_queryfile = positive_imgfiles[-1]
        
        negative_files = negative_imgfiles[:-1]
        negative_queryfile =  negative_imgfiles[-1]
        
        
        
        for idx in range(0, len(positive_imgfiles),num_per_set):
            imgs = positive_files[idx:idx+num_per_set] + negative_files[idx:idx+num_per_set]
            if len(imgs) == num_per_set*2:
                # Structure for LLaVA JSON
                json_data = {
                    "id": item['uid'] + "-" + str(idx),
                    "image": imgs + [positive_queryfile],#" |sep| ".join(imgs),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\nQuery: <image>\n" + prompts + '\nChoice list:[Positive, Negative]. Your answer is:'
                            # "value": "<image>"*len(imgs) + "\n" + prompts
                        },
                        { 
                            "from": "gpt",
                            "value": "Positive"
                        }
                    ]
                }
                json_data_list.append(json_data)
                
                imgs = np.random.choice(positive_files, size=3, replace=False).tolist() + np.random.choice(negative_files, size=3, replace=False).tolist()
                
                # breakpoint()
                json_data = {
                    "id": item['uid'] + "-" + str(idx),
                    "image": imgs + [negative_queryfile],#" |sep| ".join(imgs),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\nQuery: <image>\n" + prompts + '\nChoice list:[Positive, Negative]. Your answer is:'
                            # "value": "<image>"*len(imgs) + "\n" + prompts
                        },
                        { 
                            "from": "gpt",
                            "value": "Negative"
                        }
                    ]
                }
                json_data_list.append(json_data)
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-1.json')
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

